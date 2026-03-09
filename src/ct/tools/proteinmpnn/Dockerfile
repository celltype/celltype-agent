# ProteinMPNN — inverse folding / sequence design
# Source: https://github.com/dauparas/ProteinMPNN
# Hardware: T4 GPU, 8GB VRAM

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "torch>=2.4" \
    numpy \
    biopython

RUN git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
