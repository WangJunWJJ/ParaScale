# ParaScale Docker image

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY LICENSE .
COPY parascale/ ./parascale/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY docs/ ./docs/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

RUN mkdir -p /workspace/data /workspace/checkpoints

CMD ["/bin/bash"]
