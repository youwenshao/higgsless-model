# syntax=docker/dockerfile:1
# Applio (MIT): RVC training and inference. Upstream pins torch 2.11, which supports Blackwell.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG APPLIO_REF=main
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg curl ca-certificates libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN git clone https://github.com/IAHispano/Applio /opt/applio && git -C /opt/applio checkout ${APPLIO_REF}
RUN uv venv --python 3.11 /opt/venv \
    && uv pip install --python /opt/venv/bin/python torch==2.11.0 torchaudio==2.11.0 --index-url ${TORCH_INDEX} \
    && uv pip install --python /opt/venv/bin/python -r /opt/applio/requirements-engine.txt
WORKDIR /opt/applio
# Base models (HiFi-GAN pretraineds, RMVPE, ContentVec) are baked in so training works offline.
RUN python core.py prerequisites --pretraineds-hifigan --models --no-exe
