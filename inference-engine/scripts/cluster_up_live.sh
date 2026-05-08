#!/usr/bin/env bash
# Bring up the inference cluster for live ROS2 sensor input.
#
# Same bring-up as scripts/local_smoke.sh, minus the synthetic
# fake-publisher Deployment. Use this when you have real Raspberry
# Shake topics on the network and want the cluster ready to attach.
#
# Prerequisites: docker, kubectl, kind, helm. Run scripts/install_build_host.sh
# first on a fresh Ubuntu/Debian host if you haven't already.
#
# Before running, edit k8s/expected-sensors.yaml to list YOUR array IDs
# and the bundled-channel topics they publish on. The script aborts if
# the file still contains only the shake-001 example, since Discovery
# would otherwise sit idle waiting for a topic that doesn't exist on
# your network.
#
# Usage:
#   scripts/cluster_up_live.sh                # build, bring up, wait for Ready
#   scripts/cluster_up_live.sh --teardown     # delete the kind cluster
#   KIND_CLUSTER=other scripts/cluster_up_live.sh
#
# Once this returns, watch live results with:
#   bash scripts/tail_egress.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=_cluster_up.sh
source "$SCRIPT_DIR/_cluster_up.sh"

if [ "${1:-}" = "--teardown" ]; then
    cluster_teardown
    exit 0
fi

# Refuse to bring up the cluster if expected-sensors.yaml is still the
# unedited example. Discovery would happily run, but it would never spawn
# an Ingestor because /shake_001/data won't exist on the customer's
# network — and the silent no-op is a confusing first-run experience.
EXPECTED="$ROOT/k8s/expected-sensors.yaml"
if [ ! -f "$EXPECTED" ]; then
    echo "ERROR: $EXPECTED not found" >&2
    exit 1
fi

# Refuse if the file is still the unedited example: 'shake-001' is the
# only array key under 'arrays:'. Customers with a real shake-001 on
# their network should add a second array (or rename it) so this check
# fires only on the genuinely-untouched file.
array_keys="$(sed -E 's/#.*$//' "$EXPECTED" \
    | awk '/^[[:space:]]+arrays:[[:space:]]*$/{flag=1; next} flag && /^[[:space:]]+[a-z][a-z0-9-]*:[[:space:]]*$/{print $1}')"
if [ "$(echo "$array_keys" | grep -cvE '^\s*$')" = "1" ] \
   && echo "$array_keys" | grep -qE '^shake-001:$'; then
    cat >&2 <<EOF
ERROR: k8s/expected-sensors.yaml still contains only the 'shake-001'
example. Edit it to list your real Raspberry Shake array IDs and the
ROS2 topics they publish on, then re-run this script.

  See the file's header comments for the schema, or:
  inference-engine/README.md  →  "Discovery" section
EOF
    exit 1
fi

WITH_FAKE_PUBLISHER=0 cluster_up

cat <<EOF

=== live cluster bring-up complete ===

Discovery will spawn an Ingestor for each array in expected-sensors.yaml
as soon as that array's bundled-channel topic appears on the ROS2 graph.
Tail the spawn decision:
    kubectl logs -n $NAMESPACE -l app=discovery -f

Watch inference results in real time:
    bash scripts/tail_egress.sh

Tear down when finished:
    bash scripts/cluster_up_live.sh --teardown
EOF
