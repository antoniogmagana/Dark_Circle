#!/usr/bin/env bash
# Tail /inference_result from the running cluster and print one compact
# line per InferenceResult message. Designed for a side terminal during
# replay or live-traffic debugging.
#
# Each line:
#   HH:MM:SS.mmm sensor_id=<id> cap=<window-capture-ts> pres=<P|A>(0.NN)
#       cls=<vehicle_class>(0.NN) lat=<seconds-since-capture>
#
# Usage:
#   scripts/tail_egress.sh                  # exec into deploy/egress
#   POD=egress-abc123 scripts/tail_egress.sh
#   NAMESPACE=staging scripts/tail_egress.sh
#   TOPIC=/inference_result scripts/tail_egress.sh
#
# Prereqs: kubectl context pointed at the target cluster; egress pod
# running with the rebuilt image that includes the latency_seconds field
# on InferenceResult.
set -eo pipefail

NAMESPACE="${NAMESPACE:-default}"
TOPIC="${TOPIC:-/inference_result}"
TARGET="${POD:-deploy/egress}"

if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl not on PATH" >&2
    exit 1
fi

# Inline Python filter: parse the YAML stream emitted by `ros2 topic echo`
# and print one line per message. ros2 topic echo separates messages with
# `---`, which is the natural record boundary.
PYTHON_FILTER='
import datetime as _dt
import sys

fields = {}
def flush():
    if not fields:
        return
    cap = float(fields.get("timestamp", 0.0))
    pub = float(fields.get("publish_time", 0.0))
    lat = float(fields.get("latency_seconds", 0.0))
    sid = fields.get("sensor_id", "?").strip().strip("\"\x27")
    pres = fields.get("vehicle_detected", "false").strip().lower() == "true"
    det_conf = float(fields.get("detection_confidence", 0.0))
    cls = fields.get("vehicle_class", "").strip().strip("\"\x27") or "—"
    cls_conf = float(fields.get("classification_confidence", 0.0))
    now_str = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    pres_str = f"P({det_conf:.2f})" if pres else f"A({det_conf:.2f})"
    cls_part = f" cls={cls}({cls_conf:.2f})" if pres else ""
    lat_part = f" lat={lat * 1000:6.1f}ms" if lat > 0 else ""
    pub_part = f" pub={pub:.3f}" if pub > 0 else ""
    print(
        f"{now_str} sid={sid} cap={cap:.3f}{pub_part} {pres_str}{cls_part}{lat_part}",
        flush=True,
    )
    fields.clear()

for line in sys.stdin:
    line = line.rstrip("\n")
    if line.startswith("---"):
        flush()
        continue
    if ":" not in line or line.startswith(" "):
        continue
    key, _, val = line.partition(":")
    fields[key.strip()] = val.strip()
flush()
'

echo "=== tailing $TOPIC from $NAMESPACE/$TARGET (ctrl-c to stop) ==="
kubectl exec -n "$NAMESPACE" "$TARGET" -- /bin/bash -c "
    source /opt/ros/jazzy/setup.bash
    source /ros2_ws/install/setup.bash
    ros2 topic echo $TOPIC
" | python3 -c "$PYTHON_FILTER"
