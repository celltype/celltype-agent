# RFDiffusion — protein backbone design
# Source: https://github.com/RosettaCommons/RFdiffusion
# License: BSD 3-Clause
# Uses their official Dockerfile deps with CUDA 11.6

FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git /app/RFdiffusion

RUN pip install --no-cache-dir \
  e3nn==0.3.3 \
  hydra-core==1.3.2 \
  pyrsistent \
  pynvml \
  decorator

RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html || pip install --no-cache-dir dgl

RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger#egg=dllogger

RUN pip install --no-cache-dir /app/RFdiffusion/env/SE3Transformer && \
    pip install --no-cache-dir --no-deps /app/RFdiffusion

# Models dir — weights loaded from cache mount at runtime
RUN mkdir -p /app/RFdiffusion/models

ENV DGLBACKEND="pytorch"

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
