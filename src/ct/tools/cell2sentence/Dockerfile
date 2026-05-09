# Cell2Sentence Gemma 27B - single-cell interpretation
# Source: https://github.com/vandijklab/cell2sentence
# Setup follows the official quickstart:
#   git clone https://github.com/vandijklab/cell2sentence.git
#   conda create -n cell2sentence python=3.8
#   conda activate cell2sentence
#   make install

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

RUN git clone --depth 1 https://github.com/vandijklab/cell2sentence.git /opt/cell2sentence

RUN mamba create -n cell2sentence python=3.8 -y \
    && mamba clean --all --yes

SHELL ["/bin/bash", "-lc"]

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate cell2sentence \
    && cd /opt/cell2sentence \
    && make install \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        "torch>=2.4" \
        "transformers>=4.44,<4.49" \
        "accelerate>=0.33" \
        "sentencepiece>=0.2.0" \
        "protobuf>=5,<6" \
        "safetensors>=0.4.3" \
        "huggingface_hub>=0.24"

ENV CONDA_DEFAULT_ENV=cell2sentence
ENV CONDA_PREFIX=/opt/conda/envs/cell2sentence
ENV PATH=/opt/conda/envs/cell2sentence/bin:/opt/conda/bin:$PATH
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
ENV CELL2SENTENCE_CACHE=/root/.cache/cell2sentence
ENV CELL2SENTENCE_MODEL_ID=vandijklab/C2S-Scale-Gemma-2-27B

# Weights are downloaded on first run into a mounted cache directory.
RUN mkdir -p /root/.cache/huggingface /root/.cache/cell2sentence /workspace

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py

WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
