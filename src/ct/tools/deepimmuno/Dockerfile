FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Pin to the released tag shown in the upstream repo.
RUN git clone --depth 1 https://github.com/frankligy/DeepImmuno.git /opt/DeepImmuno

WORKDIR /opt/DeepImmuno

# Keep dependencies close to the upstream README while staying practical in a container.
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
      tensorflow==2.3.4 \
      numpy==1.18.5 \
      pandas==1.1.1 \
      scikit-learn==0.23.2 \
      scipy==1.4.1

WORKDIR /opt
COPY implementation.py /opt/implementation.py
COPY tool_entrypoint.py /opt/tool_entrypoint.py

ENTRYPOINT ["python", "/opt/tool_entrypoint.py"]
