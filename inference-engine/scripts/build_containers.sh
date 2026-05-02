#!/usr/bin/env bash
# Build container images and (optionally) load them into a kind cluster
# or push them to a registry.
#
# Build context is the inference-engine root for every image so that
# Dockerfiles can COPY src/<node>/, k8s/, ros2_interfaces/, and
# inference-protos/ without escaping the context.
#
# Usage:
#   # Build all images, default registry+tag (inference-engine/*:dev),
#   # auto-load into the dark-circle kind cluster if it exists.
#   scripts/build_containers.sh
#
#   # Subset
#   scripts/build_containers.sh discovery egress
#
#   # Build for a specific kind cluster
#   KIND_CLUSTER=mycluster scripts/build_containers.sh
#
#   # Build, tag, and push to a remote registry (Helm-chart workflow)
#   REGISTRY=registry.example.com/dark-circle TAG=v0.1.0 PUSH=1 \
#       scripts/build_containers.sh
#
#   # Build only, skip both kind load and push (CI / customer hand-off)
#   REGISTRY=registry.example.com/dark-circle TAG=v0.1.0 \
#       scripts/build_containers.sh
#
# Behavior:
#   - kind load: auto-runs when ``kind`` is on PATH AND the cluster
#     ``$KIND_CLUSTER`` (default ``dark-circle``) exists. Skipped silently
#     otherwise — useful for build-only and customer-registry workflows.
#   - docker push: only runs when ``PUSH=1`` is set. Caller is responsible
#     for ``docker login`` against the target registry beforehand.
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
#
# Default points at the current leaderboard winner on the ship metric
# `min_type_f1` (worst-case cross-location type macro-F1) per
# crl-train/saved_crl/analysis/cross_location.md:
#   multiscale/vae/v3_lowfreq, probe mlp_ztype__crl_best_aux_type
#   pres_f1=0.875, type_f1=0.6565, min_type_f1=0.4364
# The probe subdir is what carries meta.json + downstream_best.pth.
CRL_RUN_DIR="${CRL_RUN_DIR:-crl-train/saved_crl/runs/multiscale/vae/v3_lowfreq/downstream/mlp_ztype__crl_best_aux_type}"
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

if [ "${PUSH:-}" = "1" ]; then
    for name in $TARGETS; do
        image="${REGISTRY}/${name}:${TAG}"
        echo "=== pushing $image ==="
        docker push "$image"
    done
fi
