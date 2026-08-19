FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Triton / torch.compile may need a compiler at runtime
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# Install PyTorch explicitly for CUDA 12.4
RUN pip3 install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

COPY server.py /app/server.py
COPY client.py /app/client.py

EXPOSE 8000

CMD ["python3", "server.py"]
