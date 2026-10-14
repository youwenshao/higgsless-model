# syntax=docker/dockerfile:1
# ACE-Step 1.5 (MIT): repaint song sections with new lyrics (the Japanese/Korean singing path).
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG ACESTEP_REF=main
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/acestep/.venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg curl ca-certificates libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN git clone https://github.com/ace-step/ACE-Step-1.5 /opt/acestep && git -C /opt/acestep checkout ${ACESTEP_REF}
# The upstream lockfile pins torch 2.10.0+cu128 on Linux x86_64 (Blackwell-capable).
RUN cd /opt/acestep && uv sync --python 3.11
WORKDIR /opt/acestep
