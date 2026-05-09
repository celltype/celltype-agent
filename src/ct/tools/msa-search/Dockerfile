# MSA Search — Local MMseqs2 homology search
# Uses local ColabFold databases (UniRef30 + EnvDB)
# GPU-accelerated MMseqs build for fast local high-throughput search
# Databases must be mounted at /vol/colabfold_db

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip wget ca-certificates \
    && wget -qO- https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz | tar xz \
    && cp mmseqs/bin/mmseqs /usr/local/bin/ \
    && rm -rf mmseqs \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV COLABFOLD_DB=/vol/colabfold_db
ENV PYTHONUNBUFFERED=1

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
