# RhoFold+ — RNA 3D structure prediction
# Source: https://github.com/ml4bio/RhoFold (Academic)
# Hardware: A10G GPU, 16GB VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    numpy \
    biopython \
    einops

RUN git clone https://github.com/ml4bio/RhoFold.git /opt/RhoFold && \
    cd /opt/RhoFold && pip install -e .

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
