# syntax=docker/dockerfile:1
# Tools image: ffmpeg, shot detection, separation, analysis/QA scripts, GPL mixing scripts.
# Build context is the deployed worker_dir (contains worker/ and src/).
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv --python ${PYTHON_VERSION} /opt/venv \
    && uv pip install --python /opt/venv/bin/python torch torchvision torchaudio --index-url ${TORCH_INDEX}
COPY worker/docker/tools/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt
# Optional: TransNetV2 for shot detection (falls back to PySceneDetect alone when missing).
RUN uv pip install --python /opt/venv/bin/python transnetv2-pytorch \
    || echo "transnetv2-pytorch unavailable; shot detection uses PySceneDetect only"
WORKDIR /work
