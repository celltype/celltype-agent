# DiffDock — GPU-accelerated molecular docking
# Source: https://github.com/gcorso/DiffDock
# Hardware: T4/A10G GPU, 8-16GB VRAM

FROM nvcr.io/nvidia/pytorch:25.04-py3

# Install torch-geometric with CUDA-compiled extensions
# The nvidia pytorch container has torch 2.7 with CUDA 12.8
# PyG extensions must be built from source for CUDA support
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir torch-cluster torch-scatter torch-sparse torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html || \
    pip install --no-cache-dir torch-cluster torch-scatter torch-sparse torch-spline-conv
RUN pip install --no-cache-dir e3nn==0.5.1
RUN pip install --no-cache-dir rdkit biopython scipy scikit-learn pandas networkx pyyaml
RUN pip install --no-cache-dir --no-deps prody
RUN pip install --no-cache-dir --no-deps fair-esm

RUN git clone --depth 1 https://github.com/gcorso/DiffDock.git /opt/DiffDock
RUN cd /opt/DiffDock && python3 utils/precompute_series.py || true

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
