FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ARG DEBIAN_FRONTEND=noninteractive
ARG MODEL_ID=nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
ARG MODEL_DIR=/app/models/NVIDIA-NemotronLabs-VoiceChat-11B

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    MODEL_ID=${MODEL_ID} \
    MODEL_PATH=${MODEL_DIR} \
    DEVICE=cuda \
    NEMO_DIR=/opt/Speech \
    PYTHONPATH=/opt/Speech \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/opt/conda/bin:/usr/local/cuda-12.4/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    build-essential \
    ninja-build \
    cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python 3.12
RUN curl -fsSL \
      https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh \
      -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -f /tmp/miniconda.sh \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# Clone NVIDIA VoiceChat branch
RUN git clone \
    --branch nemotron-labs-voicechat \
    --depth 1 \
    https://github.com/NVIDIA-NeMo/Speech.git \
    /opt/Speech

WORKDIR /opt/Speech

# Install NeMo first.
RUN python -m pip install -e ".[all]"

# AFTER NeMo install, force the exact VoiceChat Torch stack.
# Torch is NOT included in requirements.txt.
RUN python -m pip install \
      --upgrade \
      --force-reinstall \
      torch==2.10.0 \
      torchvision==0.25.0 \
      torchaudio==2.10.0

RUN python -m pip uninstall -y nvidia-resiliency-ext || true

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Remaining VoiceChat + API dependencies.
# Torch has already been fixed above.
RUN python -m pip install \
      --no-build-isolation \
      -r /app/requirements.txt

# Verify the FINAL environment.
RUN python -c "import torch, torchvision, torchaudio, huggingface_hub; \
print('torch=', torch.__version__); \
print('torchvision=', torchvision.__version__); \
print('torchaudio=', torchaudio.__version__); \
print('torch.cuda=', torch.version.cuda); \
print('huggingface_hub=', huggingface_hub.__version__)"

# Download model — no HF token.
RUN mkdir -p ${MODEL_DIR} \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', local_dir='${MODEL_DIR}')"

COPY server.py /app/server.py

EXPOSE 8000

CMD ["python", "server.py"]
