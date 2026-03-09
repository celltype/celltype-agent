# Chai-1 — biomolecular structure prediction
# Source: https://github.com/chaidiscovery/chai-lab (Apache 2.0)
# Supports proteins, DNA, RNA, and small molecule ligands
# Hardware: A100 GPU, 40GB+ VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    chai_lab \
    biopython \
    numpy

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
