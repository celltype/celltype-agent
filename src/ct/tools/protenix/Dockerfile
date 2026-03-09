# Protenix — structure prediction outperforming AlphaFold 3
# Source: https://github.com/bytedance/Protenix (Apache 2.0)
# Hardware: A100 GPU, 40-80GB VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    protenix \
    biopython \
    numpy

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
