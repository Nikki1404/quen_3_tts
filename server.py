import asyncio
import io
import json
import subprocess
import time
import uuid
import wave
from typing import Any

import numpy as np
import torch

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Qwen3-TTS CustomVoice API",
    version="2.0.0",
)


# =============================================================================
# MODEL CONFIG
# =============================================================================

MODEL_PATH = "/app/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"

AUDIO_CHUNK_BYTES = 65536


# =============================================================================
# LOAD MODEL
# =============================================================================

print("=" * 90)
print("Qwen3-TTS CustomVoice Server")
print("=" * 90)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available inside the container."
    )

print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"Model path     : {MODEL_PATH}")

load_start = time.perf_counter()

model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

model_load_ms = (
    time.perf_counter() - load_start
) * 1000

print(
    f"Model loaded   : "
    f"{model_load_ms / 1000:.2f} s"
)


# =============================================================================
# SUPPORTED SPEAKERS / LANGUAGES
# =============================================================================

try:
    SUPPORTED_SPEAKERS = list(
        model.get_supported_speakers()
    )

    print(
        f"Speakers       : "
        f"{SUPPORTED_SPEAKERS}"
    )

except Exception:
    SUPPORTED_SPEAKERS = [
        "Aiden",
        "Ryan",
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ono_Anna",
        "Sohee",
    ]


try:
    SUPPORTED_LANGUAGES = list(
        model.get_supported_languages()
    )

    print(
        f"Languages      : "
        f"{SUPPORTED_LANGUAGES}"
    )

except Exception:
    SUPPORTED_LANGUAGES = []


# =============================================================================
# GPU INFERENCE LOCK
# =============================================================================
#
# Both WebSocket and OpenAI-compatible REST endpoint use the SAME model.
#
# This prevents two API paths from trying to run the model simultaneously
# on the same GPU.
# =============================================================================

inference_lock = asyncio.Lock()


# =============================================================================
# TIME HELPERS
# =============================================================================

def now_ns():
    return time.perf_counter_ns()


def elapsed_ms(
    start_ns,
    end_ns=None,
):
    if end_ns is None:
        end_ns = now_ns()

    return (
        end_ns - start_ns
    ) / 1_000_000.0


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def float_to_pcm16(
    wav,
):
    audio = np.asarray(
        wav,
        dtype=np.float32,
    ).squeeze()

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    return (
        audio * 32767.0
    ).astype("<i2").tobytes()


def pcm16_to_wav(
    pcm,
    sample_rate,
):
    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            pcm
        )

    return buffer.getvalue()


# =============================================================================
# SPEED FILTER
# =============================================================================

def build_atempo_filter(
    speed,
):
    """
    Build FFmpeg atempo chain.

    OpenAI-compatible speed range:
        0.25 -> 4.0
    """

    factors = []

    remaining = speed

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    if abs(
        remaining - 1.0
    ) > 0.001:

        factors.append(
            remaining
        )

    if not factors:
        return None

    return ",".join(
        f"atempo={factor}"
        for factor in factors
    )


# =============================================================================
# FORMAT ENCODING
# =============================================================================

def encode_audio(
    pcm,
    sample_rate,
    response_format,
    speed,
):

    # -------------------------------------------------------------------------
    # Raw PCM
    # -------------------------------------------------------------------------

    if (
        response_format == "pcm"
        and speed == 1.0
    ):
        return (
            pcm,
            "audio/pcm",
        )

    # -------------------------------------------------------------------------
    # Start with WAV
    # -------------------------------------------------------------------------

    wav_bytes = pcm16_to_wav(
        pcm,
        sample_rate,
    )

    if (
        response_format == "wav"
        and speed == 1.0
    ):
        return (
            wav_bytes,
            "audio/wav",
        )

    # -------------------------------------------------------------------------
    # FFmpeg conversion
    # -------------------------------------------------------------------------

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
    ]

    atempo_filter = build_atempo_filter(
        speed
    )

    if atempo_filter:
        command.extend(
            [
                "-filter:a",
                atempo_filter,
            ]
        )

    # -------------------------------------------------------------------------
    # Output format
    # -------------------------------------------------------------------------

    if response_format == "mp3":

        command.extend(
            [
                "-f",
                "mp3",
                "pipe:1",
            ]
        )

        content_type = "audio/mpeg"

    elif response_format == "wav":

        command.extend(
            [
                "-f",
                "wav",
                "pipe:1",
            ]
        )

        content_type = "audio/wav"

    elif response_format == "flac":

        command.extend(
            [
                "-f",
                "flac",
                "pipe:1",
            ]
        )

        content_type = "audio/flac"

    elif response_format == "aac":

        command.extend(
            [
                "-c:a",
                "aac",
                "-f",
                "adts",
                "pipe:1",
            ]
        )

        content_type = "audio/aac"

    elif response_format == "opus":

        command.extend(
            [
                "-c:a",
                "libopus",
                "-f",
                "opus",
                "pipe:1",
            ]
        )

        content_type = "audio/opus"

    elif response_format == "pcm":

        command.extend(
            [
                "-f",
                "s16le",
                "pipe:1",
            ]
        )

        content_type = "audio/pcm"

    else:

        raise ValueError(
            f"Unsupported response format: "
            f"{response_format}"
        )

    process = subprocess.run(
        command,
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if process.returncode != 0:

        raise RuntimeError(
            "FFmpeg encoding failed: "
            + process.stderr.decode(
                "utf-8",
                errors="ignore",
            )
        )

    return (
        process.stdout,
        content_type,
    )


# =============================================================================
# QWEN GENERATION
# =============================================================================

def generate_audio(
    text,
    language,
    speaker,
    instruct,
):

    kwargs = {
        "text": text,
        "language": language,
        "speaker": speaker,
    }

    if instruct:

        kwargs["instruct"] = instruct

    wavs, sample_rate = (
        model.generate_custom_voice(
            **kwargs
        )
    )

    if not wavs:

        raise RuntimeError(
            "Qwen3-TTS returned no audio."
        )

    return (
        wavs[0],
        int(sample_rate),
    )


# =============================================================================
# VOICE RESOLUTION
# =============================================================================

OPENAI_VOICE_MAP = {
    "alloy": "Aiden",
    "ash": "Aiden",
    "ballad": "Ryan",
    "coral": "Aiden",
    "echo": "Ryan",
    "fable": "Ryan",
    "onyx": "Ryan",
    "nova": "Aiden",
    "sage": "Ryan",
    "shimmer": "Aiden",
    "verse": "Ryan",
    "marin": "Aiden",
    "cedar": "Ryan",
}


def resolve_qwen_speaker(
    voice,
):

    # -------------------------------------------------------------------------
    # OpenAI custom voice object:
    #
    # {
    #     "id": "Aiden"
    # }
    # -------------------------------------------------------------------------

    if isinstance(
        voice,
        dict,
    ):

        voice = voice.get(
            "id",
            "Aiden",
        )

    voice = str(
        voice
    ).strip()

    # -------------------------------------------------------------------------
    # Match actual Qwen speaker
    # -------------------------------------------------------------------------

    for speaker in SUPPORTED_SPEAKERS:

        if (
            speaker.lower()
            == voice.lower()
        ):

            return speaker

    # -------------------------------------------------------------------------
    # Map standard OpenAI voice names
    # -------------------------------------------------------------------------

    mapped = OPENAI_VOICE_MAP.get(
        voice.lower()
    )

    if mapped:

        return mapped

    raise ValueError(
        f"Unsupported voice '{voice}'. "
        f"Qwen speakers: "
        f"{SUPPORTED_SPEAKERS}"
    )


# =============================================================================
# OPENAI-COMPATIBLE REQUEST
# =============================================================================

class OpenAISpeechRequest(
    BaseModel
):

    model: str = Field(
        default="qwen3-tts-0.6b",
    )

    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
    )

    voice: Any = Field(
        default="Aiden",
    )

    instructions: str = Field(
        default="",
    )

    response_format: str = Field(
        default="mp3",
    )

    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )

    # Qwen extension.
    #
    # Not required by OpenAI clients.
    #
    # Default English means an OpenAI SDK request can work
    # without sending this field.
    language: str = Field(
        default="English",
    )


# =============================================================================
# HEALTH
# =============================================================================

@app.get(
    "/health"
)
async def health():

    return {
        "status": "ok",
        "model": (
            "Qwen3-TTS-12Hz-"
            "0.6B-CustomVoice"
        ),
        "cuda": (
            torch.cuda.is_available()
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
        ),
        "model_load_ms": round(
            model_load_ms,
            2,
        ),
        "websocket_endpoint": (
            "/ws/tts"
        ),
        "openai_endpoint": (
            "/v1/audio/speech"
        ),
    }


# =============================================================================
# OPENAI-COMPATIBLE TTS
# =============================================================================

@app.post(
    "/v1/audio/speech"
)
async def openai_speech(
    request: OpenAISpeechRequest,
):

    request_received_ns = now_ns()

    request_id = str(
        uuid.uuid4()
    )

    try:

        response_format = (
            request.response_format
            .strip()
            .lower()
        )

        if response_format not in {
            "mp3",
            "opus",
            "aac",
            "flac",
            "wav",
            "pcm",
        }:

            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format must be "
                    "one of: "
                    "mp3, opus, aac, "
                    "flac, wav, pcm"
                ),
            )

        speaker = resolve_qwen_speaker(
            request.voice
        )

        # ---------------------------------------------------------------------
        # GPU inference
        # ---------------------------------------------------------------------

        inference_start_ns = (
            now_ns()
        )

        async with inference_lock:

            wav, sample_rate = (
                await asyncio.to_thread(
                    generate_audio,
                    request.input,
                    request.language,
                    speaker,
                    request.instructions,
                )
            )

        inference_end_ns = (
            now_ns()
        )

        inference_ms = elapsed_ms(
            inference_start_ns,
            inference_end_ns,
        )

        # ---------------------------------------------------------------------
        # PCM
        # ---------------------------------------------------------------------

        pcm = float_to_pcm16(
            wav
        )

        audio_duration_s = (
            len(pcm)
            / 2
            / sample_rate
        )

        # ---------------------------------------------------------------------
        # Encode requested output
        # ---------------------------------------------------------------------

        encode_start_ns = (
            now_ns()
        )

        encoded_audio, content_type = (
            await asyncio.to_thread(
                encode_audio,
                pcm,
                sample_rate,
                response_format,
                request.speed,
            )
        )

        encoding_ms = elapsed_ms(
            encode_start_ns
        )

        server_total_ms = elapsed_ms(
            request_received_ns
        )

        rtf = (
            (
                inference_ms
                / 1000.0
            )
            / audio_duration_s
            if audio_duration_s > 0
            else None
        )

        # ---------------------------------------------------------------------
        # Logs
        # ---------------------------------------------------------------------

        print()
        print("-" * 90)

        print(
            f"[OPENAI REQUEST]        "
            f"{request_id}"
        )

        print(
            f"[model]                 "
            f"{request.model}"
        )

        print(
            f"[speaker]               "
            f"{speaker}"
        )

        print(
            f"[language]              "
            f"{request.language}"
        )

        print(
            f"[format]                "
            f"{response_format}"
        )

        print(
            f"[speed]                 "
            f"{request.speed}"
        )

        print(
            f"[SERVER INFERENCE]      "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"[ENCODING]              "
            f"{encoding_ms:.2f} ms"
        )

        print(
            f"[SERVER TOTAL]          "
            f"{server_total_ms:.2f} ms"
        )

        print(
            f"[AUDIO DURATION]        "
            f"{audio_duration_s:.3f} s"
        )

        print(
            f"[RTF]                   "
            f"{rtf:.4f}"
            if rtf is not None
            else "[RTF] n/a"
        )

        print("-" * 90)

        # ---------------------------------------------------------------------
        # Chunked HTTP response
        # ---------------------------------------------------------------------

        async def audio_stream():

            for offset in range(
                0,
                len(encoded_audio),
                AUDIO_CHUNK_BYTES,
            ):

                yield encoded_audio[
                    offset:
                    offset
                    + AUDIO_CHUNK_BYTES
                ]

        return StreamingResponse(
            audio_stream(),
            media_type=content_type,
            headers={
                "X-Request-ID": (
                    request_id
                ),
                "X-Qwen-Speaker": (
                    speaker
                ),
                "X-Sample-Rate": (
                    str(sample_rate)
                ),
                "X-Server-Inference-MS": (
                    f"{inference_ms:.2f}"
                ),
                "X-Server-Encoding-MS": (
                    f"{encoding_ms:.2f}"
                ),
                "X-Server-Total-MS": (
                    f"{server_total_ms:.2f}"
                ),
                "X-Audio-Duration-S": (
                    f"{audio_duration_s:.4f}"
                ),
                "X-RTF": (
                    f"{rtf:.4f}"
                    if rtf is not None
                    else ""
                ),
            },
        )

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"[OPENAI ERROR] "
            f"request_id={request_id} "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# =============================================================================
# EXISTING WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket(
    "/ws/tts"
)
async def websocket_tts(
    websocket: WebSocket,
):

    await websocket.accept()

    print(
        f"[connected] "
        f"{websocket.client}"
    )

    try:

        while True:

            raw_message = (
                await websocket.receive_text()
            )

            request_received_ns = (
                now_ns()
            )

            request_id = str(
                uuid.uuid4()
            )

            try:

                request = json.loads(
                    raw_message
                )

                text = str(
                    request.get(
                        "text",
                        "",
                    )
                ).strip()

                language = str(
                    request.get(
                        "language",
                        "English",
                    )
                ).strip() or "English"

                speaker = str(
                    request.get(
                        "speaker",
                        "Aiden",
                    )
                ).strip() or "Aiden"

                instruct = str(
                    request.get(
                        "instruct",
                        "",
                    )
                ).strip()

                if not text:

                    raise ValueError(
                        "'text' is required."
                    )

                # =============================================================
                # ACCEPTED / SERVER TTFB
                # =============================================================

                await websocket.send_json(
                    {
                        "type": "accepted",
                        "request_id": request_id,
                        "speaker": speaker,
                        "language": language,
                    }
                )

                first_response_sent_ns = (
                    now_ns()
                )

                server_ttfb_ms = (
                    elapsed_ms(
                        request_received_ns,
                        first_response_sent_ns,
                    )
                )

                # =============================================================
                # INFERENCE
                # =============================================================

                inference_start_ns = (
                    now_ns()
                )

                async with inference_lock:

                    wav, sample_rate = (
                        await asyncio.to_thread(
                            generate_audio,
                            text,
                            language,
                            speaker,
                            instruct,
                        )
                    )

                inference_end_ns = (
                    now_ns()
                )

                inference_ms = (
                    elapsed_ms(
                        inference_start_ns,
                        inference_end_ns,
                    )
                )

                # =============================================================
                # PCM
                # =============================================================

                pcm = float_to_pcm16(
                    wav
                )

                audio_duration_s = (
                    len(pcm)
                    / 2
                    / sample_rate
                )

                # =============================================================
                # AUDIO METADATA
                # =============================================================

                await websocket.send_json(
                    {
                        "type": "audio_start",
                        "request_id": request_id,
                        "format": "pcm_s16le",
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "sample_width": 2,
                        "audio_duration_s": round(
                            audio_duration_s,
                            4,
                        ),
                        "server_ttfb_ms": round(
                            server_ttfb_ms,
                            2,
                        ),
                        "server_inference_ms": round(
                            inference_ms,
                            2,
                        ),
                    }
                )

                # =============================================================
                # AUDIO
                # =============================================================

                first_audio_sent_ns = None

                for offset in range(
                    0,
                    len(pcm),
                    AUDIO_CHUNK_BYTES,
                ):

                    await websocket.send_bytes(
                        pcm[
                            offset:
                            offset
                            + AUDIO_CHUNK_BYTES
                        ]
                    )

                    if (
                        first_audio_sent_ns
                        is None
                    ):

                        first_audio_sent_ns = (
                            now_ns()
                        )

                if (
                    first_audio_sent_ns
                    is None
                ):

                    raise RuntimeError(
                        "No PCM audio was sent."
                    )

                # =============================================================
                # METRICS
                # =============================================================

                server_ttft_ms = (
                    elapsed_ms(
                        request_received_ns,
                        first_audio_sent_ns,
                    )
                )

                server_audio_send_ms = (
                    elapsed_ms(
                        first_audio_sent_ns,
                        now_ns(),
                    )
                )

                server_total_ms = (
                    elapsed_ms(
                        request_received_ns
                    )
                )

                rtf = (
                    (
                        inference_ms
                        / 1000.0
                    )
                    / audio_duration_s
                    if audio_duration_s > 0
                    else None
                )

                # =============================================================
                # DONE
                # =============================================================

                await websocket.send_json(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "server_metrics": {
                            "ttfb_ms": round(
                                server_ttfb_ms,
                                2,
                            ),
                            "ttft_ms": round(
                                server_ttft_ms,
                                2,
                            ),
                            "first_audio_ms": round(
                                server_ttft_ms,
                                2,
                            ),
                            "inference_ms": round(
                                inference_ms,
                                2,
                            ),
                            "audio_send_ms": round(
                                server_audio_send_ms,
                                2,
                            ),
                            "total_ms": round(
                                server_total_ms,
                                2,
                            ),
                            "audio_duration_s": round(
                                audio_duration_s,
                                4,
                            ),
                            "rtf": (
                                round(
                                    rtf,
                                    4,
                                )
                                if rtf is not None
                                else None
                            ),
                        },
                    }
                )

                # =============================================================
                # LOGS
                # =============================================================

                print()
                print("-" * 90)

                print(
                    f"[request]              "
                    f"{request_id}"
                )

                print(
                    f"[speaker]              "
                    f"{speaker}"
                )

                print(
                    f"[language]             "
                    f"{language}"
                )

                print(
                    f"[text chars]           "
                    f"{len(text)}"
                )

                print(
                    f"[SERVER TTFB]          "
                    f"{server_ttfb_ms:.2f} ms"
                )

                print(
                    f"[SERVER TTFT]          "
                    f"{server_ttft_ms:.2f} ms"
                )

                print(
                    f"[SERVER FIRST AUDIO]   "
                    f"{server_ttft_ms:.2f} ms"
                )

                print(
                    f"[SERVER INFERENCE]     "
                    f"{inference_ms:.2f} ms"
                )

                print(
                    f"[SERVER AUDIO SEND]    "
                    f"{server_audio_send_ms:.2f} ms"
                )

                print(
                    f"[SERVER TOTAL]         "
                    f"{server_total_ms:.2f} ms"
                )

                print(
                    f"[AUDIO DURATION]       "
                    f"{audio_duration_s:.3f} s"
                )

                print(
                    f"[RTF]                  "
                    f"{rtf:.4f}"
                    if rtf is not None
                    else "[RTF] n/a"
                )

                print("-" * 90)

            except Exception as exc:

                print(
                    f"[error] "
                    f"request_id={request_id} "
                    f"error={exc}"
                )

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

        print(
            f"[disconnected] "
            f"{websocket.client}"
        )
