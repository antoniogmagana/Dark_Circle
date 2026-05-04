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
#   # Pick specific bundles for the inference pods
#   DETECT_BUNDLE=multiscale-vae-<run>-v1 \
#   CLASSIFY_BUNDLE=multiscale-vae-<run>-mlp_ztype-v1 \
#       scripts/build_containers.sh infer-detect infer-classify
#
# Behavior:
#   - kind load: auto-runs when ``kind`` is on PATH AND the cluster
#     ``$KIND_CLUSTER`` (default ``dark-circle``) exists. Skipped silently
#     otherwise — useful for build-only and customer-registry workflows.
#   - docker push: only runs when ``PUSH=1`` is set. Caller is responsible
#     for ``docker login`` against the target registry beforehand.
set -eo pipefail

# Hard error on the retired single-bundle env var. Silent fallthrough
# during the restructure would be a footgun.
if [ -n "${CRL_BUNDLE:-}" ]; then
    echo "ERROR: CRL_BUNDLE is no longer supported." >&2
    echo "Use DETECT_BUNDLE and CLASSIFY_BUNDLE instead." >&2
    echo "  DETECT_BUNDLE=<name>   selects from inference-engine/detect-bundles/" >&2
    echo "  CLASSIFY_BUNDLE=<name> selects from inference-engine/classify-bundles/" >&2
    exit 1
fi
if [ -n "${CRL_RUN_DIR:-}" ]; then
    echo "ERROR: CRL_RUN_DIR is no longer supported." >&2
    echo "Use DETECT_RUN_DIR and CLASSIFY_RUN_DIR instead (dev fallback path)." >&2
    exit 1
fi

REGISTRY="${REGISTRY:-inference-engine}"
TAG="${TAG:-dev}"
KIND_CLUSTER="${KIND_CLUSTER:-dark-circle}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

# Per-pod bundle selectors. Each pod is staged independently.
DETECT_BUNDLE="${DETECT_BUNDLE:-detect-default}"
CLASSIFY_BUNDLE="${CLASSIFY_BUNDLE:-classify-default}"

# Dev re-export fallback. Used when <KIND>_BUNDLE doesn't resolve to a
# bundle dir AND crl-train is installed alongside.
DETECT_RUN_DIR="${DETECT_RUN_DIR:-}"
CLASSIFY_RUN_DIR="${CLASSIFY_RUN_DIR:-}"
CRL_TRAIN_PYTHON="${CRL_TRAIN_PYTHON:-$REPO_ROOT/crl-train/.venv/bin/python}"

# stage_bundle <kind>
#
# kind = "detect" or "classify". Stages the corresponding bundle into
# build/<kind>-export/ so the matching Dockerfile can COPY it. Resolution
# order: pre-bundled (<kind>-bundles/$<KIND>_BUNDLE) first, then re-export
# from $<KIND>_RUN_DIR via crl-train.
stage_bundle() {
    local kind="$1"
    local bundle_name run_dir bundle_path out stamp resolved bundle_id

    case "$kind" in
        detect)
            bundle_name="$DETECT_BUNDLE"
            run_dir="$DETECT_RUN_DIR"
            ;;
        classify)
            bundle_name="$CLASSIFY_BUNDLE"
            run_dir="$CLASSIFY_RUN_DIR"
            ;;
        *)
            echo "stage_bundle: unknown kind '$kind' (expected detect or classify)" >&2
            return 1
            ;;
    esac

    bundle_path="$ROOT/${kind}-bundles/$bundle_name"
    out="$ROOT/build/${kind}-export"
    stamp="$out/.bundle_id"

    # ---- Path 1: pre-bundled artifact ---------------------------------
    if [ -d "$bundle_path" ]; then
        resolved="$(cd "$bundle_path" && pwd -P)"
        bundle_id="$(basename "$resolved")"

        if [ -f "$stamp" ] && \
           [ "$(cat "$stamp" 2>/dev/null)" = "$bundle_id" ] && \
           [ -z "${CRL_FORCE_EXPORT:-}" ]; then
            echo "=== ${kind} bundle '$bundle_id' already staged at $out (set CRL_FORCE_EXPORT=1 to redo) ==="
            return 0
        fi

        echo "=== copying ${kind} bundle '$bundle_id' -> build/${kind}-export ==="
        rm -rf "$out"
        mkdir -p "$out"
        cp -RL "$bundle_path"/. "$out"/
        printf "%s\n" "$bundle_id" > "$stamp"
        return 0
    fi

    # ---- Path 2: re-export from a crl-train saved run -----------------
    local kind_upper
    kind_upper="$(echo "$kind" | tr '[:lower:]' '[:upper:]')"
    if [ -z "$run_dir" ] || [ ! -d "$REPO_ROOT/$run_dir" ]; then
        echo "${kind} bundle '$bundle_name' not found at $bundle_path" >&2
        if [ -z "$run_dir" ]; then
            echo "and ${kind_upper}_RUN_DIR is not set." >&2
        else
            echo "and ${kind_upper}_RUN_DIR not found at $REPO_ROOT/$run_dir" >&2
        fi
        echo "" >&2
        echo "Set ${kind_upper}_BUNDLE to a directory under ${kind}-bundles/," >&2
        echo "or install crl-train and point ${kind_upper}_RUN_DIR at a saved run." >&2
        exit 1
    fi
    if [ ! -x "$CRL_TRAIN_PYTHON" ]; then
        echo "${kind} bundle '$bundle_name' not found at $bundle_path" >&2
        echo "and CRL_TRAIN_PYTHON not executable: $CRL_TRAIN_PYTHON" >&2
        echo "" >&2
        echo "Customer path: set ${kind_upper}_BUNDLE to a bundle under ${kind}-bundles/." >&2
        echo "Dev path:      cd $REPO_ROOT/crl-train && poetry install" >&2
        exit 1
    fi

    local run_id="run:$kind:$(basename "$run_dir")"
    if [ -f "$stamp" ] && \
       [ "$(cat "$stamp" 2>/dev/null)" = "$run_id" ] && \
       [ -z "${CRL_FORCE_EXPORT:-}" ]; then
        echo "=== ${kind} re-export for '$run_id' already staged (set CRL_FORCE_EXPORT=1 to redo) ==="
        return 0
    fi

    rm -rf "$out"
    mkdir -p "$out"
    echo "=== re-exporting ${kind} from $run_dir -> build/${kind}-export ==="
    (cd "$REPO_ROOT/crl-train" && \
        "$CRL_TRAIN_PYTHON" export_for_inference.py \
            --save-dir "$REPO_ROOT/$run_dir" \
            --bundle-kind "$kind" \
            --out-dir "$out")
    printf "%s\n" "$run_id" > "$stamp"
}

# Image short-name -> Dockerfile path lookup.
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

    # Per-pod bundle staging + label build-arg.
    build_args=()
    case "$name" in
        infer-detect)
            stage_bundle detect
            build_args+=(--build-arg "DETECT_BUNDLE_LABEL=$DETECT_BUNDLE")
            ;;
        infer-classify)
            stage_bundle classify
            build_args+=(--build-arg "CLASSIFY_BUNDLE_LABEL=$CLASSIFY_BUNDLE")
            ;;
    esac

    image="${REGISTRY}/${name}:${TAG}"
    echo "=== building $image ==="
    docker build "${build_args[@]}" -f "$dockerfile" -t "$image" .
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
