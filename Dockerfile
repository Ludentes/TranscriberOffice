FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /opt
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git \
    && python3 -m pip install -e /opt/VibeVoice

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY app ./app
COPY config.yaml .

EXPOSE 7860

CMD ["python3", "-m", "app.main"]
