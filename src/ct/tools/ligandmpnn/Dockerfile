# LigandMPNN — ligand-aware protein sequence design
# Source: https://github.com/dauparas/LigandMPNN (MIT)
# Nature Methods, March 2025
# Hardware: T4/A10G GPU, 8GB VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    numpy \
    biopython \
    prody

RUN git clone https://github.com/dauparas/LigandMPNN.git /opt/LigandMPNN && \
    cd /opt/LigandMPNN && \
    bash get_model_params.sh

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
