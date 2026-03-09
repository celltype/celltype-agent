# Evo2 — DNA sequence analysis and generation
# Source: https://github.com/ArcInstitute/evo2
# License: Apache 2.0
# Hardware: H100 GPU (compute capability >= 8.9 for FP8)
# Fully reproducible — builds from source

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install evo2 from PyPI (as per official Dockerfile)
RUN pip install evo2

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
