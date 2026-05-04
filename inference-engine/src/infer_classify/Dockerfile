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

# Source for the classification node
COPY src/infer_classify/main.py .

# CRL TorchScript bundle (encoder_*.ts + type_head_*.ts + meta.json).
# Staged into build/classify-export/ by scripts/build_containers.sh from
# the bundle named by CLASSIFY_BUNDLE (default: classify-default).
ARG CLASSIFY_BUNDLE_LABEL=unknown
LABEL com.darkcircle.classify_bundle=$CLASSIFY_BUNDLE_LABEL
COPY build/classify-export /app/model

ENV PYTHONUNBUFFERED=1
CMD ["python3", "main.py"]
