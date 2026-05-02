#!/usr/bin/env bash
# One-time sanity check that the build + cluster-up logic works.
#
# Brings up the full pipeline on a single-node kind cluster and runs
# the synthetic `fake-publisher` to drive ROS2 traffic through it. No
# real Raspberry Shake hardware required. Run this once after a fresh
# clone to confirm the build chain is healthy; thereafter, replay or
# live inference are the customer's actual workloads.
#
# Prerequisites: docker, kubectl, kind, helm.
#
# Usage:
#   scripts/local_smoke.sh                # full bring-up with fake-publisher
#   scripts/local_smoke.sh --teardown     # delete the kind cluster
#   KIND_CLUSTER=other scripts/local_smoke.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck source=_cluster_up.sh
source "$SCRIPT_DIR/_cluster_up.sh"

if [ "${1:-}" = "--teardown" ]; then
    cluster_teardown
    exit 0
fi

WITH_FAKE_PUBLISHER=1 cluster_up

cat <<EOF

=== smoke test bring-up complete ===

Tail the manager log to see the spawn decision land:
    kubectl logs -n $NAMESPACE -l app=discovery -f

Within ~10s of the fake-publisher coming up you should see:
    Spawned ingestor for shake-001

Watch inference results in real time:
    bash scripts/tail_egress.sh

When done with the smoke test:
    bash scripts/local_smoke.sh --teardown

For real work (no synthetic data):
    bash scripts/replay_in_kind.sh <audio.parquet> <seismic.parquet>   # recorded data
    # or point a real ROS2 sensor at the cluster — same pipeline, no rebuild
EOF
