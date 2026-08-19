import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
from websockets.asyncio.client import connect


# =============================================================================
# DEFAULT ENDPOINTS
# =============================================================================

DEFAULT_WS_SERVER = (
    "wss://qwen3-tts-150916788856.us-central1.run.app/ws/tts"
)

DEFAULT_OPENAI_SERVER = (
    "https://qwen3-tts-150916788856.us-central1.run.app/v1/audio/speech"
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
# AUDIO PLAYBACK
# =============================================================================

def play_pcm16(
    pcm,
    sample_rate,
    channels=1,
):
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

    print()
    print("[play] Playing audio...")

    sd.play(
        audio,
        samplerate=sample_rate,
    )

    sd.wait()

    print("[play] Finished.")


def play_wav(
    output_path,
):
    with wave.open(
        str(output_path),
        "rb",
    ) as wav_file:

        channels = wav_file.getnchannels()

        sample_rate = wav_file.getframerate()

        frames = wav_file.readframes(
            wav_file.getnframes()
        )

    play_pcm16(
        frames,
        sample_rate,
        channels,
    )


# =============================================================================
# WEBSOCKET CLIENT
# =============================================================================

async def run_websocket(
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
    # END-TO-END TIMER
    # =========================================================================

    overall_start_ns = now_ns()

    connection_start_ns = overall_start_ns

    print()
    print("=" * 80)
    print("QWEN3-TTS WEBSOCKET CLIENT")
    print("=" * 80)

    print(
        f"[connect] {server}"
    )

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

        first_response_received_ns = None
        first_audio_received_ns = None
        done_received_ns = None

        # =====================================================================
        # RECEIVE
        # =====================================================================

        async for message in websocket:

            received_ns = now_ns()

            # -----------------------------------------------------------------
            # PCM AUDIO
            # -----------------------------------------------------------------

            if isinstance(
                message,
                bytes,
            ):

                if (
                    first_audio_received_ns
                    is None
                ):

                    first_audio_received_ns = (
                        received_ns
                    )

                    request_ttfa_ms = (
                        elapsed_ms(
                            request_start_ns,
                            first_audio_received_ns,
                        )
                    )

                    e2e_ttfa_ms = elapsed_ms(
                        overall_start_ns,
                        first_audio_received_ns,
                    )

                    print(
                        f"[first-audio] "
                        f"request_ttfa="
                        f"{request_ttfa_ms:.2f} ms "
                        f"e2e_ttfa="
                        f"{e2e_ttfa_ms:.2f} ms"
                    )

                pcm.extend(
                    message
                )

                continue

            # -----------------------------------------------------------------
            # JSON EVENT
            # -----------------------------------------------------------------

            event = json.loads(
                message
            )

            event_type = event.get(
                "type"
            )

            if (
                first_response_received_ns
                is None
            ):

                first_response_received_ns = (
                    received_ns
                )

            if event_type == "accepted":

                print(
                    f"[accepted] "
                    f"request_id="
                    f"{event.get('request_id')} "
                    f"speaker="
                    f"{event.get('speaker')} "
                    f"language="
                    f"{event.get('language')}"
                )

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
                    f"sample_rate="
                    f"{sample_rate} "
                    f"audio_duration_s="
                    f"{event.get('audio_duration_s')}"
                )

            elif event_type == "done":

                done_received_ns = (
                    received_ns
                )

                server_metrics = (
                    event.get(
                        "server_metrics",
                        {},
                    )
                )

                break

            elif event_type == "error":

                raise RuntimeError(
                    event.get(
                        "error",
                        "Unknown server error",
                    )
                )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        if first_response_received_ns is None:

            raise RuntimeError(
                "No response received."
            )

        if first_audio_received_ns is None:

            raise RuntimeError(
                "No audio received."
            )

        if done_received_ns is None:

            done_received_ns = (
                now_ns()
            )

        # =====================================================================
        # METRICS
        # =====================================================================

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

        first_audio_to_done_ms = (
            elapsed_ms(
                first_audio_received_ns,
                done_received_ns,
            )
        )

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

    # =========================================================================
    # SAVE WAV
    # =========================================================================

    if not pcm:

        raise RuntimeError(
            "No audio bytes received."
        )

    if sample_rate is None:

        raise RuntimeError(
            "Sample rate not received."
        )

    output_path = Path(
        output
    )

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
    # CLIENT METRICS
    # =========================================================================

    print()
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"Connection / startup    : "
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

    print()
    print("=" * 80)
    print("END-TO-END LATENCY")
    print("=" * 80)

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
    # SERVER METRICS
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

    print()
    print(
        f"[output] "
        f"{output_path.resolve()}"
    )

    if play:

        play_pcm16(
            pcm,
            sample_rate,
            channels,
        )


# =============================================================================
# OPENAI-COMPATIBLE HTTP CLIENT
# =============================================================================

async def run_openai(
    server,
    text,
    language,
    speaker,
    instruct,
    output,
    play,
    response_format,
    speed,
):
    payload = {
        "model": "qwen3-tts-0.6b",
        "input": text,
        "voice": speaker,
        "instructions": instruct,
        "response_format": response_format,
        "speed": speed,
        "language": language,
    }

    print()
    print("=" * 80)
    print("QWEN3-TTS OPENAI-COMPATIBLE CLIENT")
    print("=" * 80)

    print(
        f"[POST] {server}"
    )

    # =========================================================================
    # TOTAL TIMER
    # =========================================================================

    overall_start_ns = now_ns()

    first_byte_ns = None

    audio_data = bytearray()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=120.0,
            read=600.0,
            write=60.0,
            pool=120.0,
        )
    ) as client:

        # =====================================================================
        # STREAM HTTP RESPONSE
        # =====================================================================

        async with client.stream(
            "POST",
            server,
            json=payload,
            headers={
                "Content-Type": (
                    "application/json"
                )
            },
        ) as response:

            headers_received_ns = (
                now_ns()
            )

            response.raise_for_status()

            print(
                f"[status] "
                f"{response.status_code}"
            )

            print(
                f"[content-type] "
                f"{response.headers.get('content-type')}"
            )

            async for chunk in (
                response.aiter_bytes()
            ):

                if not chunk:
                    continue

                if first_byte_ns is None:

                    first_byte_ns = (
                        now_ns()
                    )

                    ttfa_ms = elapsed_ms(
                        overall_start_ns,
                        first_byte_ns,
                    )

                    print(
                        f"[first-audio-byte] "
                        f"{ttfa_ms:.2f} ms"
                    )

                audio_data.extend(
                    chunk
                )

            done_ns = now_ns()

            # =================================================================
            # SERVER METRICS FROM HTTP HEADERS
            # =================================================================

            server_inference_ms = (
                response.headers.get(
                    "x-server-inference-ms"
                )
            )

            server_encoding_ms = (
                response.headers.get(
                    "x-server-encoding-ms"
                )
            )

            server_total_ms = (
                response.headers.get(
                    "x-server-total-ms"
                )
            )

            audio_duration_s = (
                response.headers.get(
                    "x-audio-duration-s"
                )
            )

            rtf = (
                response.headers.get(
                    "x-rtf"
                )
            )

            sample_rate = (
                response.headers.get(
                    "x-sample-rate"
                )
            )

    # =========================================================================
    # METRICS
    # =========================================================================

    http_ttfb_ms = elapsed_ms(
        overall_start_ns,
        headers_received_ns,
    )

    http_ttfa_ms = (
        elapsed_ms(
            overall_start_ns,
            first_byte_ns,
        )
        if first_byte_ns
        else None
    )

    http_total_ms = elapsed_ms(
        overall_start_ns,
        done_ns,
    )

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================

    output_path = Path(
        output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        bytes(audio_data)
    )

    # =========================================================================
    # PRINT METRICS
    # =========================================================================

    print()
    print("=" * 80)
    print("OPENAI-COMPATIBLE CLIENT LATENCY")
    print("=" * 80)

    print(
        f"HTTP TTFB               : "
        f"{http_ttfb_ms:.2f} ms"
    )

    print(
        f"HTTP TTFT/TTFA          : "
        f"{http_ttfa_ms:.2f} ms"
        if http_ttfa_ms is not None
        else "HTTP TTFT/TTFA          : N/A"
    )

    print(
        f"HTTP TOTAL              : "
        f"{http_total_ms:.2f} ms"
    )

    print()
    print("=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)

    print(
        f"SERVER INFERENCE        : "
        f"{server_inference_ms} ms"
    )

    print(
        f"SERVER ENCODING         : "
        f"{server_encoding_ms} ms"
    )

    print(
        f"SERVER TOTAL            : "
        f"{server_total_ms} ms"
    )

    print(
        f"AUDIO DURATION          : "
        f"{audio_duration_s} s"
    )

    print(
        f"RTF                     : "
        f"{rtf}"
    )

    print()
    print(
        f"[output] "
        f"{output_path.resolve()}"
    )

    # =========================================================================
    # PLAYBACK
    # =========================================================================

    if play:

        if response_format == "wav":

            play_wav(
                output_path
            )

        elif response_format == "pcm":

            if not sample_rate:

                raise RuntimeError(
                    "Sample rate header "
                    "missing."
                )

            play_pcm16(
                bytes(audio_data),
                int(sample_rate),
                1,
            )

        else:

            print()
            print(
                "[play-warning] "
                "Direct playback is currently "
                "supported for WAV or PCM."
            )

            print(
                "Use "
                "--response-format wav "
                "when using --play."
            )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-TTS client supporting "
            "WebSocket and OpenAI-compatible HTTP APIs"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "websocket",
            "openai",
        ],
        default="websocket",
        help=(
            "API mode: websocket or openai"
        ),
    )

    parser.add_argument(
        "--server",
        default=None,
        help=(
            "Override server URL"
        ),
    )

    parser.add_argument(
        "--text",
        required=True,
    )

    parser.add_argument(
        "--language",
        default="English",
    )

    parser.add_argument(
        "--speaker",
        default="Aiden",
    )

    parser.add_argument(
        "--instruct",
        default="",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--response-format",
        choices=[
            "wav",
            "mp3",
            "opus",
            "aac",
            "flac",
            "pcm",
        ],
        default="wav",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--play",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    if args.mode == "websocket":

        server = (
            args.server
            or DEFAULT_WS_SERVER
        )

        output = (
            args.output
            or "qwen_websocket.wav"
        )

        asyncio.run(
            run_websocket(
                server=server,
                text=args.text,
                language=args.language,
                speaker=args.speaker,
                instruct=args.instruct,
                output=output,
                play=args.play,
            )
        )

    else:

        server = (
            args.server
            or DEFAULT_OPENAI_SERVER
        )

        if args.output:

            output = args.output

        else:

            extension = (
                "raw"
                if args.response_format
                == "pcm"
                else args.response_format
            )

            output = (
                f"qwen_openai."
                f"{extension}"
            )

        asyncio.run(
            run_openai(
                server=server,
                text=args.text,
                language=args.language,
                speaker=args.speaker,
                instruct=args.instruct,
                output=output,
                play=args.play,
                response_format=(
                    args.response_format
                ),
                speed=args.speed,
            )
        )


if __name__ == "__main__":
    main()
