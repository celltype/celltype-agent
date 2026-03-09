# MSA Search — ColabFold MMseqs2 API client
# Source: Uses public ColabFold API (https://api.colabfold.com)
# No local database needed — CPU-only
# Hardware: CPU only, no GPU required

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    requests \
    biopython \
    numpy

COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]
