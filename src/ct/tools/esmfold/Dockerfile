# ESMFold — protein structure prediction from sequence
# Source: https://github.com/facebookresearch/esm
# Uses Meta's ESMFold model via HuggingFace transformers
# Hardware: H100/A100 GPU, 16GB+ VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    biopython \
    numpy

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
