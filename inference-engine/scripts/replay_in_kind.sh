#!/usr/bin/env bash
# Convenience wrapper for replay_publisher.py against a kind cluster.
# Copies the script + a paired (audio, seismic) parquet recording into
# the kind control-plane container and runs the replay there so its DDS
# participant lands on the same hostNetwork as the in-cluster pods.
#
# Usage:
#   scripts/replay_in_kind.sh <audio.parquet> <seismic.parquet> [--duration 60] [...other replay flags]
#
# Environment:
#   KIND_CLUSTER       kind cluster name (default: dark-circle)
#   FASTDDS_TRANSPORTS forces UDPv4 inside the container; matches the
#                      cluster pods. Override only if you've changed the
#                      pipeline's DDS config in values.yaml.
set -eo pipefail

KIND_CLUSTER="${KIND_CLUSTER:-dark-circle}"
NODE="${KIND_CLUSTER}-control-plane"
FASTDDS_TRANSPORTS="${FASTDDS_TRANSPORTS:-UDPv4}"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <audio.parquet> <seismic.parquet> [replay flags...]" >&2
    exit 2
fi

AUDIO="$1"; shift
SEISMIC="$1"; shift

for f in "$AUDIO" "$SEISMIC"; do
    [ -f "$f" ] || { echo "missing file: $f" >&2; exit 1; }
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Stage assets into the kind node
docker cp "$SCRIPT_DIR/replay_publisher.py" "$NODE:/tmp/replay_publisher.py"
docker cp "$AUDIO" "$NODE:/tmp/replay_audio.parquet"
docker cp "$SEISMIC" "$NODE:/tmp/replay_seismic.parquet"

# Stage the chart's ros2_interfaces source so we can colcon-build it inside
# the kind node. The replay script uses the InferenceResult + RawSensorReading
# message types; without them rclpy can't subscribe / publish.
docker cp "$ROOT/ros2_interfaces" "$NODE:/tmp/ros2_interfaces"

# One-time bootstrap inside the node: install ROS2, pyarrow, build interfaces.
# Idempotent — short-circuits if already done.
docker exec "$NODE" bash -lc '
    set -e
    if [ ! -f /tmp/.replay-bootstrap-done ]; then
        echo "=== bootstrapping replay environment ==="
        apt-get update -qq
        apt-get install -y -qq ros-jazzy-ros-base python3-pip python3-colcon-common-extensions \
            ros-jazzy-ament-cmake ros-jazzy-rosidl-default-generators \
            ros-jazzy-rosidl-default-runtime ros-jazzy-builtin-interfaces \
            ros-jazzy-std-msgs ros-jazzy-sensor-msgs >/dev/null
        pip3 install --break-system-packages pyarrow >/dev/null
        mkdir -p /tmp/ros2_ws/src
        cp -r /tmp/ros2_interfaces /tmp/ros2_ws/src/
        cd /tmp/ros2_ws
        source /opt/ros/jazzy/setup.bash
        colcon build --packages-select ros2_interfaces >/dev/null
        touch /tmp/.replay-bootstrap-done
    fi
'

# Run the replay. FASTDDS_BUILTIN_TRANSPORTS=UDPv4 matches what the cluster
# pods use (set in inference-engine-config), required for kind hostNetwork.
docker exec -it "$NODE" bash -lc "
    source /opt/ros/jazzy/setup.bash
    source /tmp/ros2_ws/install/setup.bash
    export FASTDDS_BUILTIN_TRANSPORTS=$FASTDDS_TRANSPORTS
    python3 /tmp/replay_publisher.py \\
        --audio /tmp/replay_audio.parquet \\
        --seismic /tmp/replay_seismic.parquet \\
        $*
"
