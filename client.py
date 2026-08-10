import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

from websockets.asyncio.client import connect


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    if end_ns is None:
        end_ns = now_ns()
    return (end_ns - start_ns) / 1_000_000.0


async def run_client(
    server: str,
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    output: str,
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

    # WebSocket connection/handshake latency.
    connection_start_ns = now_ns()

    print(f"[connect] {server}")

    async with connect(
        server,
        max_size=None,
        open_timeout=30,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        connected_ns = now_ns()
        connection_latency_ms = elapsed_ms(
            connection_start_ns,
            connected_ns,
        )

        # CLIENT TIMER ORIGIN:
        # Start the request latency timer immediately before sending the JSON.
        request_start_ns = now_ns()
        await websocket.send(json.dumps(payload, ensure_ascii=False))
        request_sent_ns = now_ns()

        send_call_ms = elapsed_ms(
            request_start_ns,
            request_sent_ns,
        )

        first_response_received_ns = None
        first_audio_received_ns = None
        done_received_ns = None

        async for message in websocket:
            received_ns = now_ns()

            # First binary frame = first audio frame received locally.
            if isinstance(message, bytes):
                if first_audio_received_ns is None:
                    first_audio_received_ns = received_ns

                pcm.extend(message)
                continue

            event = json.loads(message)
            event_type = event.get("type")

            # First WebSocket response from the server.
            if first_response_received_ns is None:
                first_response_received_ns = received_ns

            if event_type == "accepted":
                print(
                    f"[accepted] request_id={event.get('request_id')} "
                    f"speaker={event.get('speaker')} "
                    f"language={event.get('language')}"
                )

            elif event_type == "audio_start":
                sample_rate = int(event["sample_rate"])
                channels = int(event.get("channels", 1))
                sample_width = int(event.get("sample_width", 2))

                print(
                    f"[audio-start] sample_rate={sample_rate} "
                    f"audio_duration_s={event.get('audio_duration_s')}"
                )

            elif event_type == "done":
                done_received_ns = received_ns
                server_metrics = event.get("server_metrics", {})
                break

            elif event_type == "error":
                raise RuntimeError(event.get("error", "Unknown server error."))

        if first_response_received_ns is None:
            raise RuntimeError("No response frame received.")

        if first_audio_received_ns is None:
            raise RuntimeError("No audio frame received.")

        if done_received_ns is None:
            done_received_ns = now_ns()

        # Client-side TTFB:
        # local request start -> first WebSocket response received.
        client_ttfb_ms = elapsed_ms(
            request_start_ns,
            first_response_received_ns,
        )

        # Client-side TTFT for this TTS test:
        # local request start -> first binary PCM audio frame received.
        client_ttft_ms = elapsed_ms(
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

    if not pcm:
        raise RuntimeError("No audio received.")

    if sample_rate is None:
        raise RuntimeError("Audio metadata not received.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)

    print()
    print("=" * 90)
    print("CLIENT LATENCY")
    print("=" * 90)
    print(f"Connection latency      : {connection_latency_ms:.2f} ms")
    print(f"Client send() call      : {send_call_ms:.2f} ms")
    print(f"CLIENT TTFB             : {client_ttfb_ms:.2f} ms")
    print(f"CLIENT TTFT             : {client_ttft_ms:.2f} ms")
    print(f"CLIENT FIRST AUDIO      : {client_ttft_ms:.2f} ms")
    print(f"First audio -> done     : {first_audio_to_done_ms:.2f} ms")
    print(f"CLIENT TOTAL            : {client_total_ms:.2f} ms")

    print()
    print("=" * 90)
    print("SERVER LATENCY (REPORTED BY SERVER)")
    print("=" * 90)

    if server_metrics:
        print(f"SERVER TTFB             : {server_metrics.get('ttfb_ms')} ms")
        print(f"SERVER TTFT             : {server_metrics.get('ttft_ms')} ms")
        print(f"SERVER FIRST AUDIO      : {server_metrics.get('first_audio_ms')} ms")
        print(f"SERVER INFERENCE        : {server_metrics.get('inference_ms')} ms")
        print(f"SERVER AUDIO SEND       : {server_metrics.get('audio_send_ms')} ms")
        print(f"SERVER TOTAL            : {server_metrics.get('total_ms')} ms")
        print(f"AUDIO DURATION          : {server_metrics.get('audio_duration_s')} s")
        print(f"RTF                     : {server_metrics.get('rtf')}")
    else:
        print("No server metrics received.")

    print()
    print(f"[output] {output_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS FastAPI WebSocket latency client"
    )

    parser.add_argument(
        "--server",
        default="ws://127.0.0.1:8003/ws/tts",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="Aiden")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--output", default="output.wav")

    return parser.parse_args()


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
        )
    )


if __name__ == "__main__":
    main()
