# CUDA 12.8 **devel**（nvcc 同梱）+ Python。既定の ``uv sync`` は Sidon 中心だが、SGMSE の upfirdn2d 等（torch JIT で .cu）を入れる場合も nvcc が要るため devel を維持。
# PyTorch は uv が pyproject の cu128 インデックスから入れる（.venv）。
# NeMo / NFA は別 venv（/opt/nfa-venv）: sgmse extra が protobuf 7 系を要求するため nemo_toolkit と同一 .venv に共存できない。
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    NFA_VENV=/opt/nfa-venv \
    NFA_ALIGN_DIR=/opt/NeMo/tools/nemo_forced_aligner

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cargo \
        curl \
        git \
        python3 \
        python3-dev \
        python3-venv \
        libsndfile1 \
        rustc \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /usr/local/bin/uvx

# NeMo Forced Aligner 用: torch（cu128）+ nemo_toolkit[asr] + ツリー上の align.py（wheel に同梱されないため）
ARG NEMO_TOOLS_REF=v2.7.3
RUN set -eux \
    && python3 -m venv "${NFA_VENV}" \
    && "${NFA_VENV}/bin/pip" install --upgrade pip setuptools wheel \
    && "${NFA_VENV}/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && "${NFA_VENV}/bin/pip" install "nemo_toolkit[asr]>=2.7.3,<2.8" \
    && git clone --depth 1 --branch "${NEMO_TOOLS_REF}" https://github.com/NVIDIA/NeMo.git /opt/NeMo \
    && test -f "${NFA_ALIGN_DIR}/align.py" \
    && rm -rf /opt/NeMo/.git

ENV NFA_PYTHON="${NFA_VENV}/bin/python" \
    PATH="/workspace/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

WORKDIR /workspace
