# ImmuneBuilder — antibody, nanobody, and TCR structure prediction
# Source: https://github.com/oxpig/ImmuneBuilder

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

RUN mamba install -y -n base \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    -c bioconda \
    python=3.11 \
    pytorch \
    pytorch-cuda=12.1 \
    openmm \
    pdbfixer \
    anarci \
    libstdcxx-ng \
    pip \
    && mamba clean --all -y

RUN pip install --no-cache-dir \
    ImmuneBuilder \
    "pydantic>=2,<3"

COPY ct/tools/_schema_contract.py /opt/ct/tools/_schema_contract.py
COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/tool_entrypoint.py"]
