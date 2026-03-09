# Uni-Dock — GPU-accelerated molecular docking
# Source: https://github.com/dptech-corp/Uni-Dock (Apache 2.0)
# Hardware: T4/A10G GPU, 16GB VRAM

FROM nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    unidock \
    numpy \
    biopython

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
