#!/usr/bin/env bash
# Convenience wrapper: run replay_publisher.py against the kind smoke-test
# cluster as a one-off Pod, reusing the already-built ingestor:dev image
# (which has ROS2 Jazzy + ros2_interfaces baked in). This avoids
# installing ROS2 inside the kind node, which is minimal Debian without
# the ROS2 apt repo and would balloon the node by ~500MB.
#
# The Pod joins hostNetwork so its DDS participant lives on the same
# 172.18.0.2 IP as the in-cluster pods — the cluster cannot tell this
# apart from a real Raspberry Shake.
#
# Usage:
#   scripts/replay_in_kind.sh <audio.parquet|csv> <seismic.parquet|csv> [replay flags...]
#
# Env:
#   POD_NAME      override the one-off pod name (default: replay-publisher)
#   IMAGE         override the base image (default: inference-engine/ingestor:dev)
set -eo pipefail

POD_NAME="${POD_NAME:-replay-publisher}"
IMAGE="${IMAGE:-inference-engine/ingestor:dev}"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <audio.parquet|csv> <seismic.parquet|csv> [replay flags...]" >&2
    exit 2
fi

AUDIO="$1"; shift
SEISMIC="$1"; shift

for f in "$AUDIO" "$SEISMIC"; do
    [ -f "$f" ] || { echo "missing file: $f" >&2; exit 1; }
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Forward extra flags through the pod's argv. Argparse-friendly quoting:
# pass each as a separate `command:` element.
EXTRA_ARGS=("$@")
EXTRA_ARGS_JSON="["
first=1
for a in "${EXTRA_ARGS[@]}"; do
    if [ $first -eq 0 ]; then EXTRA_ARGS_JSON+=","; fi
    EXTRA_ARGS_JSON+="\"$a\""
    first=0
done
EXTRA_ARGS_JSON+="]"

# Clean up any prior run.
kubectl delete pod "$POD_NAME" --ignore-not-found --wait=true >/dev/null

# Stage the script + recordings inside a sleep container, then exec
# python3 against the staged files. Apply via kubectl apply -f so we
# get full control over hostNetwork and env wiring.
echo "=== creating pod $POD_NAME (image $IMAGE) ==="
kubectl apply -f - <<EOF >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  restartPolicy: Never
  containers:
    - name: replay
      image: $IMAGE
      imagePullPolicy: IfNotPresent
      command: ["sleep", "infinity"]
      env:
        - name: FASTDDS_BUILTIN_TRANSPORTS
          value: "UDPv4"
        - name: ROS_DOMAIN_ID
          valueFrom:
            configMapKeyRef:
              name: inference-engine-config
              key: ROS_DOMAIN_ID
EOF

echo "=== waiting for pod to be Ready ==="
kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=60s

# Copy the script and recordings into the pod.
echo "=== copying replay assets ==="
kubectl cp "$SCRIPT_DIR/replay_publisher.py" "$POD_NAME:/tmp/replay_publisher.py"
kubectl cp "$AUDIO"   "$POD_NAME:/tmp/replay_audio$(echo "$AUDIO"   | sed 's/.*\././')"
kubectl cp "$SEISMIC" "$POD_NAME:/tmp/replay_seismic$(echo "$SEISMIC" | sed 's/.*\././')"

# pyarrow may or may not be in the ingestor image; install if missing.
kubectl exec "$POD_NAME" -- /bin/bash -c \
    'python3 -c "import pyarrow" 2>/dev/null || pip3 install --break-system-packages --quiet pyarrow'

# Pick the actual filenames we copied (matches whatever extension the
# caller passed — .parquet or .csv).
AUDIO_FNAME="/tmp/replay_audio$(echo "$AUDIO"   | sed 's/.*\././')"
SEISMIC_FNAME="/tmp/replay_seismic$(echo "$SEISMIC" | sed 's/.*\././')"

# Run replay. The ingestor image's CMD already sources ROS2 Jazzy and
# the ros2_ws install dir, so we explicitly do the same here.
set +e
kubectl exec -t "$POD_NAME" -- /bin/bash -c "
    source /opt/ros/jazzy/setup.bash
    source /ros2_ws/install/setup.bash
    python3 /tmp/replay_publisher.py \\
        --audio $AUDIO_FNAME \\
        --seismic $SEISMIC_FNAME \\
        ${EXTRA_ARGS[*]}
"
RC=$?
set -e

echo "=== removing pod $POD_NAME ==="
kubectl delete pod "$POD_NAME" --wait=false >/dev/null

exit $RC
