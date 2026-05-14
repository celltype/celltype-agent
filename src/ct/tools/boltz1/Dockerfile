# Boltz-1 — protein / protein-ligand complex structure prediction
# Source: https://github.com/jwohlwend/boltz
# Weights auto-download on first run (~3 GB, cached at /root/.boltz)
# Hardware: A10G GPU, 16 GB+ VRAM

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip wget \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Install Boltz-1 (pinned to stable release)
RUN pip install --no-cache-dir "boltz==1.*"

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py

RUN mkdir -p /workspace
WORKDIR /workspace

ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
