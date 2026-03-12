# BindCraft — de novo protein binder design
# Source: https://github.com/martinpacesa/BindCraft
# License: MIT
# Hardware: A100/H100 GPU, 32-80GB VRAM
# Uses JAX + CUDA for AlphaFold2 backpropagation

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip git wget \
    build-essential libffi-dev libssl-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/martinpacesa/BindCraft.git /app/BindCraft

WORKDIR /app/BindCraft

RUN pip install --no-cache-dir \
    "jax[cuda12]" \
    numpy \
    pandas \
    biopython \
    scipy \
    matplotlib \
    dm-haiku \
    dm-tree \
    ml-collections \
    immutabledict \
    chex \
    optax

RUN if [ -f /app/BindCraft/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/BindCraft/requirements.txt; \
    fi

RUN pip install --no-cache-dir \
    git+https://github.com/sokrypton/ColabDesign.git

RUN git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /app/ProteinMPNN

# AF2 weights are cached on host and mounted at runtime
RUN mkdir -p /root/.cache/bindcraft /root/.cache/alphafold

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
