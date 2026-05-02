#!/usr/bin/env bash
# Shared helper: bring up the kind cluster + inference pipeline.
#
# Sourced by local_smoke.sh and replay_in_kind.sh. Not meant to be
# invoked directly. The leading underscore in the filename signals
# "private helper, not a customer entry point."
#
# Inputs (set by the caller via env or before sourcing):
#   KIND_CLUSTER    cluster name (default: dark-circle)
#   NAMESPACE       k8s namespace (default: default)
#   WITH_FAKE_PUBLISHER  "1" to deploy fake-publisher, "0" to skip
#                        (smoke=1, replay=0)
#
# Side effects:
#   - Creates the kind cluster if absent (Calico CNI, podSubnet 192.168.0.0/16).
#   - Installs KEDA in the keda namespace if absent.
#   - Builds the six container images and loads them into the kind node.
#   - Applies all k8s manifests in dependency order.
#   - Waits for core deployments to be Ready before returning.
#
# Idempotent: re-runs against an existing cluster pick up any new
# manifests without disturbing healthy pods.
set -eo pipefail

# Resolve repo paths relative to this helper's location, not the
# caller's. SCRIPT_DIR points at scripts/, ROOT at the inference-engine
# checkout root.
_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$_HELPER_DIR"
ROOT="$(cd "$_HELPER_DIR/.." && pwd)"

KIND_CLUSTER="${KIND_CLUSTER:-dark-circle}"
NAMESPACE="${NAMESPACE:-default}"
WITH_FAKE_PUBLISHER="${WITH_FAKE_PUBLISHER:-0}"

cluster_up() {
    cd "$ROOT"

    # --- 0. Tool checks -----------------------------------------------------
    for tool in docker kubectl kind helm; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "missing required tool: $tool" >&2
            return 1
        fi
    done
    docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable" >&2; return 1; }

    # --- 1. kind cluster ----------------------------------------------------
    if kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER"; then
        echo "=== kind cluster '$KIND_CLUSTER' already exists; reusing ==="
    else
        echo "=== creating kind cluster '$KIND_CLUSTER' (Calico CNI) ==="
        kind create cluster --name "$KIND_CLUSTER" --config "$SCRIPT_DIR/kind/kind-linux.yaml"
        echo "=== installing Calico for multicast-capable pod networking ==="
        # --server-side: the tigera-operator CRDs exceed the 262144-byte
        # annotation limit that client-side `kubectl apply` enforces on
        # newer kubectl versions, so the apply has to happen server-side.
        kubectl apply --server-side -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/tigera-operator.yaml

        # The Installation CR depends on the installations.operator.tigera.io
        # CRD just applied above. The kube-apiserver registers new CRDs
        # asynchronously: `kubectl wait --for=established` returns as soon
        # as the CRD object hits the Established condition, but the
        # apiserver's REST-mapping discovery cache may still not list the
        # kind for several seconds after that. The CR apply hits "no
        # matches for kind Installation" if it loses the race. Poll until
        # `kubectl api-resources` actually lists Installation, not just
        # until the CRD object is happy.
        echo "  waiting for Installation kind to register with apiserver..."
        for i in $(seq 1 60); do
            if kubectl api-resources --api-group=operator.tigera.io 2>/dev/null \
                    | grep -q '^installations\b'; then
                break
            fi
            sleep 2
        done
        cat <<EOF | kubectl apply -f -
apiVersion: operator.tigera.io/v1
kind: Installation
metadata:
  name: default
spec:
  calicoNetwork:
    ipPools:
      - blockSize: 26
        cidr: 192.168.0.0/16
        encapsulation: VXLAN
        natOutgoing: Enabled
        nodeSelector: all()
EOF
        kubectl wait --for=condition=available --timeout=300s \
            deployment/calico-kube-controllers -n calico-system || true
    fi
    kubectl config use-context "kind-$KIND_CLUSTER"

    # --- 2. KEDA ------------------------------------------------------------
    if ! kubectl get ns keda >/dev/null 2>&1; then
        echo "=== installing KEDA ==="
        helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
        helm repo update >/dev/null
        helm install keda kedacore/keda --namespace keda --create-namespace
    fi
    kubectl wait --for=condition=available --timeout=180s deployment/keda-operator -n keda || true

    # --- 3. Build & load images --------------------------------------------
    "$SCRIPT_DIR/build_containers.sh"

    # --- 4. Apply manifests in dependency order ----------------------------
    echo "=== applying manifests ==="
    kubectl apply -f k8s/nats/nats-deployment.yaml
    kubectl apply -f k8s/nats/nats-service.yaml
    kubectl wait --for=condition=available --timeout=180s deployment/nats -n "$NAMESPACE"

    kubectl apply -f k8s/nats/jetstream-streams.yaml
    kubectl wait --for=condition=complete --timeout=120s job/jetstream-init -n "$NAMESPACE"

    kubectl apply -f k8s/rbac/
    kubectl apply -f k8s/inference-engine-config.yaml
    kubectl apply -f k8s/sensor-config.yaml
    kubectl apply -f k8s/expected-sensors.yaml
    kubectl apply -f k8s/discovery.yaml
    if [ "$WITH_FAKE_PUBLISHER" = "1" ]; then
        kubectl apply -f k8s/fake-publisher.yaml
    else
        # If fake-publisher was deployed by an earlier run (e.g. a prior
        # smoke test) and we're now in replay/live mode, take it down so
        # synthetic data doesn't compete with real input on the same
        # ROS2 topics.
        kubectl delete deployment fake-publisher -n "$NAMESPACE" --ignore-not-found >/dev/null
    fi
    kubectl apply -f k8s/infer-detect.yaml
    kubectl apply -f k8s/infer-classify.yaml
    kubectl apply -f k8s/egress.yaml
    kubectl apply -f k8s/keda/

    # --- 5. Wait for core pods ---------------------------------------------
    echo "=== waiting for core pods ==="
    kubectl wait --for=condition=available --timeout=180s deployment/discovery -n "$NAMESPACE"
    if [ "$WITH_FAKE_PUBLISHER" = "1" ]; then
        kubectl wait --for=condition=available --timeout=180s deployment/fake-publisher -n "$NAMESPACE"
    fi
    kubectl wait --for=condition=available --timeout=180s deployment/infer-detect -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=180s deployment/infer-classify -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=180s deployment/egress -n "$NAMESPACE"

    echo
    echo "=== current pods ==="
    kubectl get pods -n "$NAMESPACE"
}

cluster_teardown() {
    echo "=== deleting kind cluster '$KIND_CLUSTER' ==="
    kind delete cluster --name "$KIND_CLUSTER" || true
}
