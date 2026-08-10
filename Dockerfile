FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirement.txt .

RUN pip install -r requirement.txt

RUN mkdir -p /app/models && \
    huggingface-cli download \
    Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --local-dir /app/models/Qwen3-TTS-12Hz-1.7B-CustomVoice

COPY server.py .
COPY client.py .

EXPOSE 8880

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
