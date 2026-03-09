# GNINA — deep learning molecular docking
# Source: https://github.com/gnina/gnina (Apache 2.0)
# CNN-based scoring, v1.3 (2025)
# Hardware: T4 GPU, 8GB VRAM

FROM nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip wget libopenbabel-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/gnina/gnina/releases/latest/download/gnina -O /usr/local/bin/gnina && \
    chmod +x /usr/local/bin/gnina

RUN pip3 install --no-cache-dir numpy biopython

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
