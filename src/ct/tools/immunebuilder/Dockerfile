# ImmuneBuilder — Antibody/TCR/Nanobody structure prediction
FROM continuumio/miniconda3:latest

# 1. Force Python 3.10 for stable ML environment
RUN conda install -y python=3.10

# 2. Install biological dependencies via conda
RUN conda install -y -c bioconda anarci && \
    conda install -y -c conda-forge openmm pdbfixer && \
    conda clean -afy

# 3. Install PyTorch (CPU) and ImmuneBuilder
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ImmuneBuilder

# 4. PRE-FETCH WEIGHTS: Instant runtime execution
RUN python -c "from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2, TCRBuilder2; ABodyBuilder2(); NanoBodyBuilder2(); TCRBuilder2()"

# 5. CellType standard tool execution pattern
COPY tool_entrypoint.py /opt/tool_entrypoint.py
COPY implementation.py /opt/implementation.py

RUN mkdir -p /workspace /vol/workspace
WORKDIR /workspace

ENTRYPOINT ["python3", "/opt/tool_entrypoint.py"]