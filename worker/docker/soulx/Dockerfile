# syntax=docker/dockerfile:1
# SoulX-Singer (Apache-2.0): singing voice synthesis, SVC and the transcription preprocess.
# Upstream pins torch 2.2 for inference; preprocess uses 2.10, which also runs on Blackwell (sm_120).
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG SOULX_REF=main
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/soulx
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg curl ca-certificates libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN git clone https://github.com/Soul-AILab/SoulX-Singer /opt/soulx && git -C /opt/soulx checkout ${SOULX_REF}
RUN uv venv --python 3.11 /opt/venv \
    && grep -viE '^(torch|torchaudio|torchcodec|sageattention)\b' /opt/soulx/requirements.txt > /tmp/req-main.txt \
    && grep -viE '^(torch|torchaudio|torchcodec|sageattention)\b' /opt/soulx/preprocess/requirements.txt > /tmp/req-pre.txt \
    && uv pip install --python /opt/venv/bin/python -r /tmp/req-main.txt \
    && uv pip install --python /opt/venv/bin/python -r /tmp/req-pre.txt \
    && uv pip install --python /opt/venv/bin/python torch==2.10.0 torchaudio==2.10.0 --index-url ${TORCH_INDEX} \
    && uv pip install --python /opt/venv/bin/python torchcodec huggingface_hub
WORKDIR /opt/soulx
