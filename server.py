import asyncio
import json
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from qwen_tts import Qwen3TTSModel


app = FastAPI(title="Qwen3-TTS CustomVoice WebSocket Server")

MODEL_PATH = "/app/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
AUDIO_CHUNK_BYTES = 65536

print("=" * 90)
print("Qwen3-TTS CustomVoice WebSocket Server")
print("=" * 90)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available inside the container.")

print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"Model path     : {MODEL_PATH}")

load_start = time.perf_counter()

model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

print(f"Model loaded   : {(time.perf_counter() - load_start):.2f} s")

try:
    print(f"Speakers       : {model.get_supported_speakers()}")
    print(f"Languages      : {model.get_supported_languages()}")
except Exception:
    pass

# One model / one GPU: keep inference serialized for predictable benchmarking.
inference_lock = asyncio.Lock()


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    if end_ns is None:
        end_ns = now_ns()
    return (end_ns - start_ns) / 1_000_000.0


def float_to_pcm16(wav) -> bytes:
    audio = np.asarray(wav, dtype=np.float32).squeeze()
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def generate_audio(text: str, language: str, speaker: str, instruct: str):
    kwargs = {
        "text": text,
        "language": language,
        "speaker": speaker,
    }
    if instruct:
        kwargs["instruct"] = instruct

    wavs, sample_rate = model.generate_custom_voice(**kwargs)

    if not wavs:
        raise RuntimeError("Qwen3-TTS returned no audio.")

    return wavs[0], int(sample_rate)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    print(f"[connected] {websocket.client}")

    try:
        while True:
            raw_message = await websocket.receive_text()

            # SERVER TIMER ORIGIN:
            # Start the request timer as soon as the complete JSON request has
            # arrived in the FastAPI application.
            request_received_ns = now_ns()
            request_id = str(uuid.uuid4())

            try:
                request = json.loads(raw_message)

                text = str(request.get("text", "")).strip()
                language = str(request.get("language", "English")).strip() or "English"
                speaker = str(request.get("speaker", "Aiden")).strip() or "Aiden"
                instruct = str(request.get("instruct", "")).strip()

                if not text:
                    raise ValueError("'text' is required.")

                # First server response frame.
                await websocket.send_json(
                    {
                        "type": "accepted",
                        "request_id": request_id,
                        "speaker": speaker,
                        "language": language,
                    }
                )
                first_response_sent_ns = now_ns()
                server_ttfb_ms = elapsed_ms(
                    request_received_ns,
                    first_response_sent_ns,
                )

                inference_start_ns = now_ns()

                async with inference_lock:
                    wav, sample_rate = await asyncio.to_thread(
                        generate_audio,
                        text,
                        language,
                        speaker,
                        instruct,
                    )

                inference_end_ns = now_ns()
                inference_ms = elapsed_ms(
                    inference_start_ns,
                    inference_end_ns,
                )

                pcm = float_to_pcm16(wav)
                audio_duration_s = len(pcm) / (2 * sample_rate)

                # Metadata is sent immediately before audio.
                await websocket.send_json(
                    {
                        "type": "audio_start",
                        "request_id": request_id,
                        "format": "pcm_s16le",
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "sample_width": 2,
                        "audio_duration_s": round(audio_duration_s, 4),
                        "server_ttfb_ms": round(server_ttfb_ms, 2),
                        "server_inference_ms": round(inference_ms, 2),
                    }
                )

                first_audio_sent_ns = None

                for offset in range(0, len(pcm), AUDIO_CHUNK_BYTES):
                    await websocket.send_bytes(
                        pcm[offset : offset + AUDIO_CHUNK_BYTES]
                    )

                    if first_audio_sent_ns is None:
                        first_audio_sent_ns = now_ns()

                if first_audio_sent_ns is None:
                    raise RuntimeError("No PCM audio was sent.")

                # Here TTFT means "time to first audio frame" for this TTS benchmark.
                server_ttft_ms = elapsed_ms(
                    request_received_ns,
                    first_audio_sent_ns,
                )

                server_audio_send_ms = elapsed_ms(
                    first_audio_sent_ns,
                    now_ns(),
                )

                server_total_ms = elapsed_ms(request_received_ns)
                rtf = (
                    (inference_ms / 1000.0) / audio_duration_s
                    if audio_duration_s > 0
                    else None
                )

                await websocket.send_json(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "server_metrics": {
                            "ttfb_ms": round(server_ttfb_ms, 2),
                            "ttft_ms": round(server_ttft_ms, 2),
                            "first_audio_ms": round(server_ttft_ms, 2),
                            "inference_ms": round(inference_ms, 2),
                            "audio_send_ms": round(server_audio_send_ms, 2),
                            "total_ms": round(server_total_ms, 2),
                            "audio_duration_s": round(audio_duration_s, 4),
                            "rtf": round(rtf, 4) if rtf is not None else None,
                        },
                    }
                )

                print()
                print("-" * 90)
                print(f"[request]              {request_id}")
                print(f"[speaker]              {speaker}")
                print(f"[language]             {language}")
                print(f"[text chars]           {len(text)}")
                print(f"[SERVER TTFB]          {server_ttfb_ms:.2f} ms")
                print(f"[SERVER TTFT]          {server_ttft_ms:.2f} ms")
                print(f"[SERVER FIRST AUDIO]   {server_ttft_ms:.2f} ms")
                print(f"[SERVER INFERENCE]     {inference_ms:.2f} ms")
                print(f"[SERVER AUDIO SEND]    {server_audio_send_ms:.2f} ms")
                print(f"[SERVER TOTAL]         {server_total_ms:.2f} ms")
                print(f"[AUDIO DURATION]       {audio_duration_s:.3f} s")
                print(f"[RTF]                  {rtf:.4f}" if rtf is not None else "[RTF] n/a")
                print("-" * 90)

            except Exception as exc:
                print(f"[error] request_id={request_id} error={exc}")
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "error": str(exc),
                        }
                    )
                except Exception:
                    break

    except WebSocketDisconnect:
        print(f"[disconnected] {websocket.client}")
