# OpenFold3 — protein structure prediction
# Source: https://github.com/aqlaboratory/openfold-3
# License: Apache 2.0
# Hardware: A100/H100 GPU
# Fully reproducible — builds from source, no local dependencies

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget libopenmpi-dev libaio-dev git python3-dbg build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone OpenFold3
RUN git clone --depth 1 https://github.com/aqlaboratory/openfold-3.git /opt/openfold3

# Install conda environment
RUN mamba env create -f /opt/openfold3/environments/production-linux-64.yml --name openfold3 \
    && mamba clean --all --yes

ENV PATH=/opt/conda/envs/openfold3/bin:$PATH
ENV CONDA_PREFIX=/opt/conda/envs/openfold3
ENV CONDA_DEFAULT_ENV=openfold3

# Install third-party deps (CUTLASS for DeepSpeed)
WORKDIR /opt
RUN bash /opt/openfold3/scripts/install_third_party_dependencies.sh
ENV CUTLASS_PATH=/opt/cutlass
ENV KMP_AFFINITY=none
ENV LIBRARY_PATH=/opt/conda/envs/openfold3/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/openfold3/lib:$LD_LIBRARY_PATH

# Install OpenFold3 package
WORKDIR /opt/openfold3
RUN uv pip install --no-deps --editable .

# Weights downloaded at runtime to mounted cache (/root/.openfold3)

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
