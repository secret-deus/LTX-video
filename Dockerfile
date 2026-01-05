FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-pip python3-venv \
    ffmpeg \
    pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev \
    libavfilter-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) PyTorch CUDA (cu121)
RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

# 2) Clone official repo
COPY LTX-Video/ /workspace/LTX-Video
WORKDIR /workspace/LTX-Video

# 3) Install official package editable
RUN pip3 install -e .

# 4) Install webui deps
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

# 5) Copy gradio app
COPY app_gradio.py /workspace/LTX-Video/app_gradio.py

EXPOSE 7860

CMD ["python3", "app_gradio.py", "--server_name", "0.0.0.0", "--server_port", "7860"]