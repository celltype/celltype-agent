# REINVENT 4 — de novo molecular design with RL
# Source: https://github.com/MolecularAI/REINVENT4 (Apache 2.0, AstraZeneca)
# Hardware: A10G GPU, 16GB VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    reinvent \
    rdkit \
    numpy \
    toml

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
