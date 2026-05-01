#!/usr/bin/env bash
# Build local container images and load them into a kind cluster.
#
# Build context is the inference-engine root for every image so that
# Dockerfiles can COPY src/<node>/, k8s/, ros2_interfaces/, and
# inference-protos/ without escaping the context.
#
# Usage:
#   scripts/build_containers.sh                  # discovery, ingestor, fake-publisher
#   scripts/build_containers.sh discovery        # subset
#   KIND_CLUSTER=mycluster scripts/build_containers.sh
#
# Skips `kind load` if `kind` is not on PATH or no cluster exists, so the
# script is also useful for plain docker builds against a remote registry
# (set REGISTRY=foo to tag images as foo/<name>:dev instead of the default
# inference-engine/<name>:dev — push is up to you).
set -eo pipefail

REGISTRY="${REGISTRY:-inference-engine}"
TAG="${TAG:-dev}"
KIND_CLUSTER="${KIND_CLUSTER:-dark-circle}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

# Where the CRL run lives, relative to the repo root. Override by setting
# CRL_RUN_DIR before invoking the script. The export script writes
# encoder_*.ts, type_head_*.ts, and meta.json into build/crl-export/.
CRL_RUN_DIR="${CRL_RUN_DIR:-crl-train/saved_crl/runs/multiscale/vae/v1}"
CRL_TRAIN_PYTHON="${CRL_TRAIN_PYTHON:-$REPO_ROOT/crl-train/.venv/bin/python}"

# Stage the CRL TorchScript export into build/crl-export/ so the
# infer-detect / infer-classify Dockerfiles can COPY it. Re-runs every
# build so a checkpoint or export-script update is picked up. Skipped if
# the export already exists and CRL_FORCE_EXPORT is not set.
stage_crl_export() {
    local out="$ROOT/build/crl-export"
    if [ -d "$out" ] && [ -z "${CRL_FORCE_EXPORT:-}" ]; then
        echo "=== CRL export already staged at $out (set CRL_FORCE_EXPORT=1 to redo) ==="
        return 0
    fi
    if [ ! -d "$REPO_ROOT/$CRL_RUN_DIR" ]; then
        echo "CRL run dir not found: $REPO_ROOT/$CRL_RUN_DIR" >&2
        exit 1
    fi
    if [ ! -x "$CRL_TRAIN_PYTHON" ]; then
        echo "CRL_TRAIN_PYTHON not executable: $CRL_TRAIN_PYTHON" >&2
        echo "(install crl-train deps with: cd $REPO_ROOT/crl-train && poetry install)" >&2
        exit 1
    fi
    rm -rf "$out"
    mkdir -p "$out"
    echo "=== exporting CRL run $CRL_RUN_DIR -> build/crl-export ==="
    (cd "$REPO_ROOT/crl-train" && \
        "$CRL_TRAIN_PYTHON" export_for_inference.py \
            --save-dir "$REPO_ROOT/$CRL_RUN_DIR" \
            --out-dir "$out")
}

# Image short-name -> Dockerfile path lookup.
# Plain function instead of an associative array, since macOS still ships
# bash 3.2 which has no `declare -A`.
dockerfile_for() {
    case "$1" in
        discovery)      echo "src/discovery/Dockerfile" ;;
        ingestor)       echo "src/ingestor/Dockerfile" ;;
        fake-publisher) echo "src/fake_publisher/Dockerfile" ;;
        infer-detect)   echo "src/infer_detect/Dockerfile" ;;
        infer-classify) echo "src/infer_classify/Dockerfile" ;;
        egress)         echo "src/egress/Dockerfile" ;;
        *)              return 1 ;;
    esac
}

ALL_TARGETS="discovery ingestor fake-publisher infer-detect infer-classify egress"

if [ "$#" -gt 0 ]; then
    TARGETS="$*"
else
    TARGETS="$ALL_TARGETS"
fi

for name in $TARGETS; do
    dockerfile=$(dockerfile_for "$name") || {
        echo "unknown image: $name (valid: $ALL_TARGETS)" >&2
        exit 1
    }

    # Stage the CRL TorchScript export for the inference images. Both
    # infer-detect and infer-classify consume the same bundle; the export
    # only runs once per build_containers.sh invocation.
    case "$name" in
        infer-detect|infer-classify)
            stage_crl_export
            ;;
    esac

    image="${REGISTRY}/${name}:${TAG}"
    echo "=== building $image ==="
    docker build -f "$dockerfile" -t "$image" .
done

if command -v kind >/dev/null 2>&1; then
    if kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER"; then
        for name in $TARGETS; do
            image="${REGISTRY}/${name}:${TAG}"
            echo "=== loading $image into kind cluster '$KIND_CLUSTER' ==="
            kind load docker-image "$image" --name "$KIND_CLUSTER"
        done
    else
        echo "kind cluster '$KIND_CLUSTER' not found; skipping kind load" >&2
        echo "(create it with: kind create cluster --name $KIND_CLUSTER)" >&2
    fi
else
    echo "kind not on PATH; skipping kind load" >&2
fi
