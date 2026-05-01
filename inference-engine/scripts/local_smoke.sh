#!/usr/bin/env bash
# End-to-end smoke test on a Linux kind cluster (the canonical local target;
# see README §DDS for macOS-specific guidance). Brings up:
#
#   - kind cluster (Calico CNI for multicast-friendly DDS)
#   - KEDA (autoscaler for the inference + egress pods)
#   - NATS + JetStream + the three streams we use
#   - RBAC, ConfigMaps, Discovery, fake publisher, inference, egress
#
# Then waits for the pipeline to spin up and points the user at the right
# logs.
#
# Prerequisites: docker, kubectl, kind, helm.
#
# Usage:
#   scripts/local_smoke.sh                # full bring-up
#   scripts/local_smoke.sh --teardown     # delete the kind cluster
#   KIND_CLUSTER=other scripts/local_smoke.sh
set -eo pipefail

KIND_CLUSTER="${KIND_CLUSTER:-dark-circle}"
NAMESPACE="${NAMESPACE:-default}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [ "${1:-}" = "--teardown" ]; then
    echo "=== deleting kind cluster '$KIND_CLUSTER' ==="
    kind delete cluster --name "$KIND_CLUSTER" || true
    exit 0
fi

# --- 0. Tool checks ---------------------------------------------------------
for tool in docker kubectl kind helm; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "missing required tool: $tool" >&2
        exit 1
    fi
done
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable" >&2; exit 1; }

# --- 1. kind cluster --------------------------------------------------------
if kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER"; then
    echo "=== kind cluster '$KIND_CLUSTER' already exists; reusing ==="
else
    echo "=== creating kind cluster '$KIND_CLUSTER' (Calico CNI) ==="
    kind create cluster --name "$KIND_CLUSTER" --config "$SCRIPT_DIR/kind/kind-linux.yaml"
    echo "=== installing Calico for multicast-capable pod networking ==="
    kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/tigera-operator.yaml
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

# --- 2. KEDA ----------------------------------------------------------------
if ! kubectl get ns keda >/dev/null 2>&1; then
    echo "=== installing KEDA ==="
    helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
    helm repo update >/dev/null
    helm install keda kedacore/keda --namespace keda --create-namespace
fi
kubectl wait --for=condition=available --timeout=180s deployment/keda-operator -n keda || true

# --- 3. Build & load images -------------------------------------------------
"$SCRIPT_DIR/build_containers.sh"

# --- 4. Apply manifests in dependency order ---------------------------------
echo "=== applying manifests ==="
kubectl apply -f k8s/nats/nats-deployment.yaml
kubectl apply -f k8s/nats/nats-service.yaml
kubectl wait --for=condition=available --timeout=180s deployment/nats -n "$NAMESPACE"

kubectl apply -f k8s/nats/jetstream-streams.yaml
kubectl wait --for=condition=complete --timeout=120s job/jetstream-init -n "$NAMESPACE"

kubectl apply -f k8s/rbac/
kubectl apply -f k8s/sensor-config.yaml
kubectl apply -f k8s/expected-sensors.yaml
kubectl apply -f k8s/discovery.yaml
kubectl apply -f k8s/fake-publisher.yaml
kubectl apply -f k8s/infer-detect.yaml
kubectl apply -f k8s/infer-classify.yaml
kubectl apply -f k8s/egress.yaml
kubectl apply -f k8s/keda/

# --- 5. Wait for core pods --------------------------------------------------
echo "=== waiting for core pods ==="
kubectl wait --for=condition=available --timeout=180s deployment/discovery -n "$NAMESPACE"
kubectl wait --for=condition=available --timeout=180s deployment/fake-publisher -n "$NAMESPACE"
kubectl wait --for=condition=available --timeout=180s deployment/infer-detect -n "$NAMESPACE"
kubectl wait --for=condition=available --timeout=180s deployment/infer-classify -n "$NAMESPACE"
kubectl wait --for=condition=available --timeout=180s deployment/egress -n "$NAMESPACE"

echo
echo "=== current pods ==="
kubectl get pods -n "$NAMESPACE"

# --- 6. Show what to look for next -----------------------------------------
cat <<EOF

=== next steps ===

Tail the manager log to see the spawn decision land:
    kubectl logs -n $NAMESPACE -l app=discovery -f

Within ~10s of the fake-publisher coming up you should see:
    Spawned ingestor for shake-001

Inspect each JetStream consumer to confirm messages are flowing:
    kubectl exec -n $NAMESPACE deploy/nats -- \\
        nats --server=localhost:4222 consumer info SENSOR_DATA infer-detect
    kubectl exec -n $NAMESPACE deploy/nats -- \\
        nats --server=localhost:4222 consumer info DETECTION_RESULT infer-classify
    kubectl exec -n $NAMESPACE deploy/nats -- \\
        nats --server=localhost:4222 consumer info DETECTION_RESULT egress-detection
    kubectl exec -n $NAMESPACE deploy/nats -- \\
        nats --server=localhost:4222 consumer info CLASSIFICATION_RESULT egress-classification

Watch the ROS2 output:
    kubectl exec -n $NAMESPACE deploy/egress -- /bin/bash -c \\
        'source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash \\
         && ros2 topic echo /inference_result'

Test teardown — remove an array from the ConfigMap and watch the manager:
    kubectl edit configmap expected-sensors -n $NAMESPACE
    kubectl logs -n $NAMESPACE -l app=discovery -f
    # 3 polls (~15s) later the Ingestor should be torn down.

When done:
    scripts/local_smoke.sh --teardown
EOF
