# Qwen3-TTS CustomVoice — FastAPI WebSocket + EC2 GPU + Latency Metrics

Model:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The model is downloaded **during `docker build`** and stored inside the Docker image.

The EC2 container runs FastAPI with Uvicorn:

```text
ws://<EC2-IP>:8003/ws/tts
```

## Metrics

### Server

```text
SERVER TTFB
request received by FastAPI
→ accepted WebSocket response sent

SERVER TTFT / FIRST AUDIO
request received by FastAPI
→ first binary PCM audio frame sent

SERVER INFERENCE
generate_custom_voice() start
→ generate_custom_voice() returns waveform

SERVER AUDIO SEND
first audio binary frame sent
→ all audio frames sent

SERVER TOTAL
request received
→ completion processing

RTF
inference seconds / generated audio seconds
```

### Client

```text
Connection latency
start WebSocket connect
→ WebSocket connection established

CLIENT TTFB
request start
→ first WebSocket response received

CLIENT TTFT / FIRST AUDIO
request start
→ first binary PCM audio frame received

CLIENT TOTAL
request start
→ done response received
```

For this TTS benchmark, `TTFT` is being used as **time to first transmitted/received audio frame**.

Strictly speaking, "time to first token" is a model-token metric. The public
`generate_custom_voice()` API returns the completed waveform and does not expose
the first internal acoustic token through this application.

Therefore the useful end-user speech metric here is effectively TTFA:
**time to first audio**.

## Build on EC2

Verify GPU:

```bash
nvidia-smi
```

Verify Docker GPU runtime:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi
```

Build:

```bash
docker build -t qwen3-tts .
```

The Hugging Face model downloads during this build.

Run:

```bash
docker run -d \
  --gpus all \
  --restart unless-stopped \
  --name qwen3-tts \
  -p 8003:8003 \
  qwen3-tts
```

Logs:

```bash
docker logs -f qwen3-tts
```

Health:

```bash
curl http://localhost:8003/health
```

## EC2 Security Group

Allow TCP `8880` from your laptop's public IP.

## Local client

You only need `client.py` locally.

```bash
python3 -m venv client_env
source client_env/bin/activate
pip install "websockets>=15,<18"
```

Then:

```bash
python client.py \
  --server ws://<EC2_PUBLIC_IP>:8880/ws/tts \
  --text "I cannot believe we finally made it!" \
  --language English \
  --speaker Aiden \
  --instruct "Speak happily and with excitement." \
  --output excited.wav
```

Example output:

```text
[connect] ws://54.x.x.x:8003/ws/tts
[accepted] request_id=... speaker=Aiden language=English
[audio-start] sample_rate=... audio_duration_s=...

==========================================================================================
CLIENT LATENCY
==========================================================================================
Connection latency      : 79.11 ms
Client send() call      : 0.15 ms
CLIENT TTFB             : 40.52 ms
CLIENT TTFT             : 1318.43 ms
CLIENT FIRST AUDIO      : 1318.43 ms
First audio -> done     : 7.82 ms
CLIENT TOTAL            : 1326.25 ms

==========================================================================================
SERVER LATENCY (REPORTED BY SERVER)
==========================================================================================
SERVER TTFB             : 0.13 ms
SERVER TTFT             : 1238.54 ms
SERVER FIRST AUDIO      : 1238.54 ms
SERVER INFERENCE        : 1234.80 ms
SERVER AUDIO SEND       : 1.81 ms
SERVER TOTAL            : 1240.43 ms
AUDIO DURATION          : 3.52 s
RTF                     : 0.3508
```

The difference between client and server measurements includes network transport,
WebSocket framing, scheduling, and any client/server-side overhead around the
timed regions.
