FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages

COPY requirement.txt .

RUN pip3 install --break-system-packages -r requirement.txt

RUN mkdir -p /app/models && \
    huggingface-cli download \
    Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --local-dir /app/models/Qwen3-TTS-12Hz-1.7B-CustomVoice

COPY server.py .
COPY client.py .

EXPOSE 8003

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
