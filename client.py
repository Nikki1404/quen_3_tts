import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from websockets.asyncio.client import connect


# =============================================================================
# DEFAULT CLOUD RUN WEBSOCKET URL
# =============================================================================

DEFAULT_SERVER = (
    "wss://qwen3-tts-150916788856.us-central1.run.app/ws/tts"
)


# =============================================================================
# TIME HELPERS
# =============================================================================

def now_ns():
    return time.perf_counter_ns()


def elapsed_ms(start_ns, end_ns=None):
    if end_ns is None:
        end_ns = now_ns()

    return (end_ns - start_ns) / 1_000_000.0


# =============================================================================
# CLIENT
# =============================================================================

async def run_client(
    server,
    text,
    language,
    speaker,
    instruct,
    output,
    play,
):

    payload = {
        "text": text,
        "language": language,
        "speaker": speaker,
        "instruct": instruct,
    }

    pcm = bytearray()

    sample_rate = None
    channels = 1
    sample_width = 2

    server_metrics = {}

    # =========================================================================
    # END-TO-END TIMER STARTS BEFORE WEBSOCKET CONNECTION
    # =========================================================================

    overall_start_ns = now_ns()

    connection_start_ns = overall_start_ns

    print()
    print("=" * 80)
    print("QWEN3-TTS CLIENT")
    print("=" * 80)

    print(f"[connect] {server}")

    # =========================================================================
    # WEBSOCKET CONNECTION
    # =========================================================================

    async with connect(
        server,
        max_size=None,
        open_timeout=120,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
    ) as websocket:

        connected_ns = now_ns()

        connection_latency_ms = elapsed_ms(
            connection_start_ns,
            connected_ns,
        )

        print(
            f"[connection-latency] "
            f"{connection_latency_ms:.2f} ms"
        )

        # =====================================================================
        # SEND REQUEST
        # =====================================================================

        request_start_ns = now_ns()

        await websocket.send(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

        request_sent_ns = now_ns()

        send_call_ms = elapsed_ms(
            request_start_ns,
            request_sent_ns,
        )

        # =====================================================================
        # RESPONSE TIMESTAMPS
        # =====================================================================

        first_response_received_ns = None
        first_audio_received_ns = None
        done_received_ns = None

        # =====================================================================
        # RECEIVE SERVER EVENTS / AUDIO
        # =====================================================================

        async for message in websocket:

            received_ns = now_ns()

            # -----------------------------------------------------------------
            # BINARY MESSAGE = PCM AUDIO
            # -----------------------------------------------------------------

            if isinstance(message, bytes):

                if first_audio_received_ns is None:

                    first_audio_received_ns = received_ns

                    request_ttfa_ms = elapsed_ms(
                        request_start_ns,
                        first_audio_received_ns,
                    )

                    e2e_ttfa_ms = elapsed_ms(
                        overall_start_ns,
                        first_audio_received_ns,
                    )

                    print(
                        f"[first-audio] "
                        f"request_ttfa={request_ttfa_ms:.2f} ms "
                        f"e2e_ttfa={e2e_ttfa_ms:.2f} ms"
                    )

                pcm.extend(message)

                continue

            # -----------------------------------------------------------------
            # JSON CONTROL MESSAGE
            # -----------------------------------------------------------------

            event = json.loads(message)

            event_type = event.get("type")

            if first_response_received_ns is None:
                first_response_received_ns = received_ns

            # -----------------------------------------------------------------
            # ACCEPTED
            # -----------------------------------------------------------------

            if event_type == "accepted":

                print(
                    f"[accepted] "
                    f"request_id={event.get('request_id')} "
                    f"speaker={event.get('speaker')} "
                    f"language={event.get('language')}"
                )

            # -----------------------------------------------------------------
            # AUDIO START
            # -----------------------------------------------------------------

            elif event_type == "audio_start":

                sample_rate = int(
                    event["sample_rate"]
                )

                channels = int(
                    event.get(
                        "channels",
                        1,
                    )
                )

                sample_width = int(
                    event.get(
                        "sample_width",
                        2,
                    )
                )

                print(
                    f"[audio-start] "
                    f"sample_rate={sample_rate} "
                    f"audio_duration_s="
                    f"{event.get('audio_duration_s')}"
                )

            # -----------------------------------------------------------------
            # DONE
            # -----------------------------------------------------------------

            elif event_type == "done":

                done_received_ns = received_ns

                server_metrics = event.get(
                    "server_metrics",
                    {},
                )

                break

            # -----------------------------------------------------------------
            # ERROR
            # -----------------------------------------------------------------

            elif event_type == "error":

                raise RuntimeError(
                    event.get(
                        "error",
                        "Unknown server error",
                    )
                )

        # =====================================================================
        # VALIDATE RESPONSE
        # =====================================================================

        if first_response_received_ns is None:

            raise RuntimeError(
                "No response received from server."
            )

        if first_audio_received_ns is None:

            raise RuntimeError(
                "No audio received from server."
            )

        if done_received_ns is None:

            done_received_ns = now_ns()

        # =====================================================================
        # CLIENT REQUEST METRICS
        #
        # These start AFTER WebSocket connection.
        # =========================================================================

        client_ttfb_ms = elapsed_ms(
            request_start_ns,
            first_response_received_ns,
        )

        client_ttfa_ms = elapsed_ms(
            request_start_ns,
            first_audio_received_ns,
        )

        client_total_ms = elapsed_ms(
            request_start_ns,
            done_received_ns,
        )

        first_audio_to_done_ms = elapsed_ms(
            first_audio_received_ns,
            done_received_ns,
        )

        # =====================================================================
        # END-TO-END METRICS
        #
        # These start BEFORE WebSocket connection.
        #
        # Therefore cold-start / Cloud Run startup delay occurring during
        # connection establishment is included.
        # =====================================================================

        e2e_ttfb_ms = elapsed_ms(
            overall_start_ns,
            first_response_received_ns,
        )

        e2e_ttfa_ms = elapsed_ms(
            overall_start_ns,
            first_audio_received_ns,
        )

        e2e_total_ms = elapsed_ms(
            overall_start_ns,
            done_received_ns,
        )

        connection_to_first_response_ms = elapsed_ms(
            connected_ns,
            first_response_received_ns,
        )

        connection_to_first_audio_ms = elapsed_ms(
            connected_ns,
            first_audio_received_ns,
        )

    # =========================================================================
    # AUDIO VALIDATION
    # =========================================================================

    if not pcm:

        raise RuntimeError(
            "No audio bytes received."
        )

    if sample_rate is None:

        raise RuntimeError(
            "Sample rate was not received."
        )

    # =========================================================================
    # SAVE WAV
    # =========================================================================

    output_path = Path(output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            channels
        )

        wav_file.setsampwidth(
            sample_width
        )

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            pcm
        )

    # =========================================================================
    # CLIENT REQUEST LATENCY
    # =========================================================================

    print()
    print("=" * 80)
    print("CLIENT REQUEST LATENCY")
    print("=" * 80)

    print(
        f"Connection latency      : "
        f"{connection_latency_ms:.2f} ms"
    )

    print(
        f"Send call               : "
        f"{send_call_ms:.2f} ms"
    )

    print(
        f"CLIENT TTFB             : "
        f"{client_ttfb_ms:.2f} ms"
    )

    print(
        f"CLIENT TTFT/TTFA        : "
        f"{client_ttfa_ms:.2f} ms"
    )

    print(
        f"Audio -> Done           : "
        f"{first_audio_to_done_ms:.2f} ms"
    )

    print(
        f"CLIENT TOTAL            : "
        f"{client_total_ms:.2f} ms"
    )

    # =========================================================================
    # END-TO-END / COLD START OBSERVATION
    # =========================================================================

    print()
    print("=" * 80)
    print("END-TO-END / COLD START OBSERVATION")
    print("=" * 80)

    print(
        f"Connection / startup    : "
        f"{connection_latency_ms:.2f} ms"
    )

    print(
        f"Connection -> response  : "
        f"{connection_to_first_response_ms:.2f} ms"
    )

    print(
        f"Connection -> audio     : "
        f"{connection_to_first_audio_ms:.2f} ms"
    )

    print(
        f"E2E TTFB                : "
        f"{e2e_ttfb_ms:.2f} ms"
    )

    print(
        f"E2E TTFT/TTFA           : "
        f"{e2e_ttfa_ms:.2f} ms"
    )

    print(
        f"E2E TOTAL               : "
        f"{e2e_total_ms:.2f} ms"
    )

    # =========================================================================
    # SERVER LATENCY
    # =========================================================================

    print()
    print("=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)

    if server_metrics:

        print(
            f"SERVER TTFB             : "
            f"{server_metrics.get('ttfb_ms')} ms"
        )

        print(
            f"SERVER TTFT/TTFA        : "
            f"{server_metrics.get('ttft_ms')} ms"
        )

        print(
            f"SERVER FIRST AUDIO      : "
            f"{server_metrics.get('first_audio_ms')} ms"
        )

        print(
            f"SERVER INFERENCE        : "
            f"{server_metrics.get('inference_ms')} ms"
        )

        print(
            f"SERVER AUDIO SEND       : "
            f"{server_metrics.get('audio_send_ms')} ms"
        )

        print(
            f"SERVER TOTAL            : "
            f"{server_metrics.get('total_ms')} ms"
        )

        print(
            f"AUDIO DURATION          : "
            f"{server_metrics.get('audio_duration_s')} s"
        )

        print(
            f"RTF                     : "
            f"{server_metrics.get('rtf')}"
        )

    else:

        print(
            "No server metrics received."
        )

    # =========================================================================
    # COLD START INTERPRETATION
    # =========================================================================

    print()
    print("=" * 80)
    print("COLD START INTERPRETATION")
    print("=" * 80)

    print(
        "Connection/startup includes network + TLS + WebSocket handshake "
        "+ possible Cloud Run cold-start delay."
    )

    print(
        "E2E TTFA includes connection/startup + request processing "
        "+ inference until first audio."
    )

    print(
        "Compare this run against an immediate second warm run to "
        "estimate cold-start overhead."
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    print()
    print(
        f"[output] "
        f"{output_path.resolve()}"
    )

    # =========================================================================
    # OPTIONAL PLAYBACK
    # =========================================================================

    if play:

        print()
        print(
            "[play] Playing generated audio..."
        )

        audio = np.frombuffer(
            pcm,
            dtype=np.int16,
        )

        if channels > 1:

            audio = audio.reshape(
                -1,
                channels,
            )

        audio = (
            audio.astype(np.float32)
            / 32768.0
        )

        sd.play(
            audio,
            samplerate=sample_rate,
        )

        sd.wait()

        print(
            "[play] Finished."
        )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-TTS Cloud Run WebSocket client "
            "with latency, cold-start observation, "
            "and optional playback"
        )
    )

    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=(
            "WebSocket server URL. "
            f"Default: {DEFAULT_SERVER}"
        ),
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize",
    )

    parser.add_argument(
        "--language",
        default="English",
        help="Language",
    )

    parser.add_argument(
        "--speaker",
        default="Aiden",
        help="Speaker name",
    )

    parser.add_argument(
        "--instruct",
        default="",
        help="Voice instruction",
    )

    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file",
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help="Play generated audio locally",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    asyncio.run(
        run_client(
            server=args.server,
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            output=args.output,
            play=args.play,
        )
    )


if __name__ == "__main__":
    main()
