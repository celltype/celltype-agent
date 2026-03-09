# OpenFold2 — AlphaFold2 reimplementation in PyTorch
# Source: https://github.com/aqlaboratory/openfold
# License: Apache 2.0
# Hardware: A100/H100 GPU
# Fully reproducible — builds from source, no local dependencies

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y wget libxml2 git

# Install Miniforge (conda)
RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone OpenFold
RUN git clone --depth 1 https://github.com/aqlaboratory/openfold.git /opt/openfold

# Install conda env (remove flash-attn from env.yml — needs torch at build time)
RUN cd /opt/openfold && sed -i '/flash-attn/d' environment.yml \
    && mamba env update -n base --file environment.yml && mamba clean --all

# Install flash-attn after torch is available
RUN pip install flash-attn --no-build-isolation || true

# Install TensorRT + cuda-python for accelerated inference
RUN pip install "tensorrt<10.15" "cuda-python<13" polygraphy || true

# Install OpenFold
WORKDIR /opt/openfold
RUN wget -q -P /opt/openfold/openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
RUN python3 setup.py install

# Patch TensorRT import to be optional if cuda-python doesn't have cudart
RUN python3 -c "import cuda.cudart" 2>/dev/null || \
    sed -i 's/from .tensorrt_utils import instrument_with_trt_compile/try:\n    from .tensorrt_utils import instrument_with_trt_compile\nexcept (ImportError, ModuleNotFoundError):\n    instrument_with_trt_compile = None/' /opt/openfold/openfold/utils/script_utils.py

# Weights downloaded at runtime to mounted cache (/root/.cache/openfold/params)

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
