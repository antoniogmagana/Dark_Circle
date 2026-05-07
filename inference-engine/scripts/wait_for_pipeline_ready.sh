#!/usr/bin/env bash
# Wait for the inference pipeline to become Ready, then print a banner.
#
# Cluster-agnostic: works on kind, EKS, GKE, on-prem — anywhere the chart
# is installed. Uses kubectl wait so it respects readiness probes set on
# infer-detect / infer-classify / egress, which themselves only flip Ready
# after the model is loaded and the NATS subscription is bound.
#
# Usage:
#   scripts/wait_for_pipeline_ready.sh [--namespace <ns>] [--timeout <seconds>]
#
# Defaults: namespace=default, timeout=300s.
set -eo pipefail

NAMESPACE="default"
TIMEOUT="300"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --namespace|-n) NAMESPACE="$2"; shift 2 ;;
        --timeout|-t)   TIMEOUT="$2";   shift 2 ;;
        --help|-h)
            sed -n '2,12p' "$0" >&2
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: $0 [--namespace <ns>] [--timeout <seconds>]" >&2
            exit 2
            ;;
    esac
done

DEPLOYS=("infer-detect" "infer-classify" "egress")

echo "Waiting for pipeline (namespace=$NAMESPACE, timeout=${TIMEOUT}s)..."
failed=()
for d in "${DEPLOYS[@]}"; do
    if ! kubectl wait --for=condition=available --timeout="${TIMEOUT}s" \
            "deployment/$d" -n "$NAMESPACE" >/dev/null 2>&1; then
        failed+=("$d")
    fi
done

if [ "${#failed[@]}" -eq 0 ]; then
    echo
    echo "==========================================="
    echo "Pipeline ready. You can now publish messages."
    echo "==========================================="
    exit 0
fi

echo >&2
echo "ERROR: the following deployments did not become Available within ${TIMEOUT}s:" >&2
for d in "${failed[@]}"; do
    echo "  - $d" >&2
done
echo >&2
echo "Diagnose with:" >&2
for d in "${failed[@]}"; do
    echo "  kubectl describe deployment/$d -n $NAMESPACE" >&2
    echo "  kubectl logs -l app=$d -n $NAMESPACE --tail=50" >&2
done
exit 1
