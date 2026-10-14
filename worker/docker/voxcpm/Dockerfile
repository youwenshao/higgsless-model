# syntax=docker/dockerfile:1
# VoxCPM2 (Apache-2.0): design an invented voice from a text description.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python torch torchaudio --index-url ${TORCH_INDEX} \
    && uv pip install --python /opt/venv/bin/python voxcpm soundfile huggingface_hub
WORKDIR /work
