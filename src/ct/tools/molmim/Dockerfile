# MolMIM — molecule optimization using RDKit
# Uses matched molecular pairs and property-guided mutations
# Hardware: CPU (RDKit is CPU-based)

FROM python:3.11-slim

RUN pip install --no-cache-dir rdkit numpy

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
