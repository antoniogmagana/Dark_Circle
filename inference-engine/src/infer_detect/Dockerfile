FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the inference-engine root.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        nats-py \
        numpy \
        torch

# Shared proto package
COPY inference-protos /inference-protos
RUN pip install --no-cache-dir /inference-protos

# Source for the detection node
COPY src/infer_detect/main.py .

# CRL TorchScript bundle (encoder_*.ts + meta.json). Staged into
# build/crl-export/ by scripts/build_containers.sh from a saved CRL run.
# Override at runtime by mounting a different directory at /app/model.
COPY build/crl-export /app/model

ENV PYTHONUNBUFFERED=1
CMD ["python3", "main.py"]
