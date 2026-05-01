#!/usr/bin/env bash
# One-shot setup script for the replay_publisher.py customer test tool.
#
# What this installs (Ubuntu / Debian only):
#   - ROS2 Jazzy base   (apt: ros-jazzy-ros-base + colcon)
#   - Python pyarrow    (pip; needed only when replaying .parquet files)
#   - ros2_interfaces   (the inference-engine custom message package,
#                        built from this repo's ros2_interfaces/ directory)
#
# What it does NOT do:
#   - Configure your ROS_DOMAIN_ID, RMW, or DDS profiles. Replay inherits
#     whatever ROS2 environment you've sourced before invoking it.
#
# After this script completes, source the workspace and run replay:
#
#   source /opt/ros/jazzy/setup.bash
#   source ~/ros2_replay_ws/install/setup.bash
#   python3 inference-engine/scripts/replay_publisher.py --audio ... --seismic ...
#
# DIY: see chart/README.md "Testing on pre-recorded data" for the
# manual install steps if your platform isn't Ubuntu / Debian.
set -eo pipefail

WORKSPACE="${WORKSPACE:-$HOME/ros2_replay_ws}"

# ---- 0. Platform check ------------------------------------------------------
if [ ! -f /etc/os-release ]; then
    echo "ERROR: cannot detect OS; this script supports Ubuntu / Debian only." >&2
    echo "       See chart/README.md for manual install instructions." >&2
    exit 1
fi
. /etc/os-release
case "$ID" in
    ubuntu|debian) ;;
    *)
        echo "ERROR: $PRETTY_NAME is not supported by this script." >&2
        echo "       This installer is Ubuntu / Debian only. For other" >&2
        echo "       platforms, follow the manual steps in chart/README.md" >&2
        echo "       'Testing on pre-recorded data'." >&2
        exit 1
        ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "ERROR: this script needs root or sudo. Re-run as root or install sudo." >&2
        exit 1
    fi
    SUDO=sudo
fi

# ---- 1. ROS2 Jazzy ----------------------------------------------------------
if ! dpkg -s ros-jazzy-ros-base >/dev/null 2>&1; then
    echo "=== installing ROS2 Jazzy ==="
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq curl gnupg lsb-release software-properties-common

    $SUDO install -d /usr/share/keyrings
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | $SUDO gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $UBUNTU_CODENAME main" \
        | $SUDO tee /etc/apt/sources.list.d/ros2.list >/dev/null

    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq \
        ros-jazzy-ros-base \
        python3-colcon-common-extensions \
        ros-jazzy-ament-cmake \
        ros-jazzy-rosidl-default-generators \
        ros-jazzy-rosidl-default-runtime \
        ros-jazzy-builtin-interfaces \
        ros-jazzy-std-msgs
else
    echo "=== ROS2 Jazzy already installed ==="
fi

# ---- 2. Python deps ---------------------------------------------------------
echo "=== installing Python dependencies ==="
# pyarrow only — pandas etc. not needed; CSV uses the stdlib.
# --break-system-packages: safe on Ubuntu 24.04 where pip refuses to
# touch /usr/lib/python without it. Customers preferring a venv can skip
# this script and follow the DIY steps in chart/README.md.
python3 -m pip install --break-system-packages --quiet pyarrow

# ---- 3. Build ros2_interfaces ----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -d "$ROOT/ros2_interfaces" ]; then
    echo "ERROR: $ROOT/ros2_interfaces not found. Run this script from inside" >&2
    echo "       a clone of the inference-engine repository." >&2
    exit 1
fi

echo "=== building ros2_interfaces in $WORKSPACE ==="
mkdir -p "$WORKSPACE/src"
# Use cp -R so we don't symlink across user boundaries (colcon prefers
# a real source checkout under src/).
rm -rf "$WORKSPACE/src/ros2_interfaces"
cp -R "$ROOT/ros2_interfaces" "$WORKSPACE/src/ros2_interfaces"

(
    cd "$WORKSPACE"
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    colcon build --packages-select ros2_interfaces
)

# ---- 4. Done ----------------------------------------------------------------
cat <<EOF

=== install_replay.sh complete ===

To run the replay tool, source ROS2 and the workspace, then invoke:

    source /opt/ros/jazzy/setup.bash
    source $WORKSPACE/install/setup.bash
    python3 $ROOT/scripts/replay_publisher.py \\
        --audio <path/to/audio.parquet|csv> \\
        --seismic <path/to/seismic.parquet|csv> \\
        --duration 60

For options and CSV column mapping see:
    python3 $ROOT/scripts/replay_publisher.py --help

EOF
