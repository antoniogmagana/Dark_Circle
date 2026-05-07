# Inference Engine: Expected Sensor-Pair Whitelist

**Status:** Draft
**Date:** 2026-04-30
**Owner:** Brandon Taylor

## Context

Today the Discovery node spawns Ingestor pods reactively: it polls the ROS2
graph every 5 s, groups any topics matching `RawSensorReading` by URL prefix,
and creates one Ingestor Deployment per discovered array
(`src/discovery/main.py:47-67`). Channel roles (audio / seismic / accel x/y/z)
are inferred inside the Ingestor by string-matching the trailing token of
`msg.sensor_id` against a hardcoded `CHANNEL_MAP`
(`src/ingestor/main.py:12-18, 47-54`).

This works for the demo fleet but has three problems we want to fix now,
before adding more sensor sites:

1. **No expectation contract.** The system has no notion of which sensors
   *should* be present. A half-online array (audio up, seismic down) gets a
   live Ingestor that silently never produces output, because the
   buffer's window-flush is acoustic-triggered.
2. **Convention-locked.** Roles are derived from substring matching
   (`aud`, `ehz`, `ene`...). Renaming a publisher topic breaks the pipeline
   silently.
3. **No whitelist.** Any new `RawSensorReading` topic on the network spawns
   an Ingestor. Useful for plug-and-play, dangerous for a production fleet.

We are moving to an **explicit-pairs model**: an operator-curated dictionary
of expected arrays, mounted into the Discovery pod as a ConfigMap. Discovery
becomes the gatekeeper — it spawns an Ingestor only when *all required
topics* for a configured array are visible on the ROS2 graph, and it injects
explicit topic→role mappings into the Ingestor so role binding is
configuration, not convention.

## Out of Scope

- Hot-reconfiguring a running Ingestor (no restart-on-accel-arrival).
- Replacing NATS or any downstream pipeline component.
- Auto-detecting sensor health beyond "topic present on graph."
- A control-plane API for runtime mutation of the dictionary; ConfigMap edits
  are the only supported workflow.

## Configuration Format

Mounted at `/etc/inference-engine/expected-sensors.yaml` inside the Discovery
pod. Schema:

```yaml
arrays:
  shake-001:                     # array id: lowercase RFC 1123 subdomain
    audio: /shake_001/aud        # topic strings keep firmware naming
    seismic: /shake_001/ehz
    accel:                       # optional
      x: /shake_001/ene
      y: /shake_001/enn
      z: /shake_001/enz
  shake-002:
    audio: /shake_002/aud
    seismic: /shake_002/ehz
    # no accel block -> accel not required, not subscribed
```

**Array ID constraints.** Each top-level YAML key becomes a Kubernetes
Deployment name (``ingestor-<array_id>``) and must satisfy the RFC 1123
subdomain rule: lowercase letters, digits, and hyphens; start with a
letter; end alphanumeric; no underscores; at most ``63 - len("ingestor-")
= 54`` characters. The topic *values* can be anything ROS2 allows — they
reflect the firmware's published topic names.

**Required keys per array:** `audio`, `seismic`.
**Optional key:** `accel` — if present, must contain all three of `x`, `y`, `z`.
A partial accel block (e.g., only `x` and `y`) is a config error and the
manager refuses to consider that array until fixed.

## Manager Behavior

### Definition of "complete"

An array is **complete** on a given poll iff every topic listed in its
config entry is currently visible on the ROS2 graph as a
`ros2_interfaces/msg/RawSensorReading` publisher. If `accel` is absent from
the config, the three accel topics are not required.

If `accel` is present in the config, all three of `accel.x`, `accel.y`,
`accel.z` must be visible — partial accel never spawns.

### Spawn policy

On each 5 s poll:
1. Re-read the ConfigMap from disk (mounted ConfigMaps are
   eventually-consistent; cheap re-read avoids inotify/SIGHUP).
2. For each array in the config that is **not** currently active:
   - If complete → spawn the Ingestor.
   - If incomplete → do nothing. Log "awaiting `<array>`: missing
     `<topics>`" only when the missing-topic set changes (state-change log,
     not per-tick spam).

### Teardown policy (mirrors current grace-poll behavior)

An Ingestor is torn down when **either**:
- Its array entry has been removed from the ConfigMap for `GRACE_POLLS=3`
  consecutive polls (≈15 s), OR
- One or more of its required topics has been absent from the graph for
  `GRACE_POLLS=3` consecutive polls.

Both conditions feed the same `absent_counts` counter so config edits and
sensor outages are absorbed identically.

### Unknown-topic handling

A topic that does not belong to any array in the config is logged at INFO
once per state change ("ignoring unconfigured topic `<topic>`") and
otherwise ignored. No fall-back prefix-grouping behavior.

## Ingestor Changes

The Ingestor's subscription model changes from "subscribe by topic, infer
role from message content" to "subscribe by topic, role bound at
subscription time."

### New env vars (replace `SENSOR_TOPICS`)

- `SENSOR_ROLE_MAP` — JSON object, e.g.
  `{"acoustic":"/shake_001/aud","seismic":"/shake_001/ehz",
    "accel_x":"/shake_001/ene","accel_y":"/shake_001/enn",
    "accel_z":"/shake_001/enz"}`. Keys use the buffer-side role names
  (`acoustic`, `seismic`, `accel_x/y/z`); values are ROS2 topic strings.
- `SENSOR_ARRAY` — unchanged.
- `NATS_URL` — unchanged.

### Subscription dispatch

`IngestorNode.__init__` iterates the role map and creates one
`create_subscription` per (topic, role) pair. Each subscription's callback
is bound to the role at construction time — either via `functools.partial`
or a per-topic closure. The shared callback signature becomes
`listener_callback(msg, role)` and writes directly to
`buffer.load_buffer(role, ...)`. The `msg.sensor_id`-suffix path and the
top-of-file `CHANNEL_MAP` are deleted.

Rationale: with the manager already knowing each topic's role, recomputing
it from the message body is duplicate state that drifts. Binding role at
subscription time also means the Ingestor no longer cares whether a
publisher renames itself, only whether the configured topic delivers data.

## Critical Files

- `inference-engine/src/discovery/main.py` — substantial rewrite of
  `_get_sensor_arrays`, `_spawn`, `_teardown`, `_poll`, plus a new
  `_load_config()` helper.
- `inference-engine/src/ingestor/main.py` — replace `SENSOR_TOPICS` parsing
  with `SENSOR_ROLE_MAP` JSON parsing; rewrite subscription loop; remove
  `CHANNEL_MAP` and the suffix-derivation in `listener_callback`.
- `inference-engine/k8s/ingestor-template.yaml` — replace the
  `SENSOR_TOPICS` env entry with `SENSOR_ROLE_MAP`. New placeholder
  `<sensor_role_map_json>`.
- `inference-engine/k8s/discovery.yaml` — add a ConfigMap volume mount for
  `expected-sensors.yaml` at `/etc/inference-engine/`.
- `inference-engine/k8s/expected-sensors.yaml` — **new** ConfigMap manifest
  with an example two-array fleet.
- `inference-engine/tests/test_discovery.py` — substantial rewrite. The
  current 13 tests assume prefix-grouping discovery; most need to be
  rewritten around the config-driven completeness check.
- `inference-engine/tests/test_ingestor.py` — update channel-mapping tests
  (currently 2 of the 17) to reflect role-map subscription dispatch
  instead of `CHANNEL_MAP` suffix lookup.
- `inference-engine/tests/conftest.py` — extend `mock_ros2_node` to support
  the new "topic missing" scenarios used by completeness tests.

## Verification

1. **Unit:** new tests for `_load_config()` (valid/invalid YAML, partial
   accel block rejection), `_is_complete()`, and the state-change log
   throttle.
2. **Integration:** existing pytest suite must reach 100% pass with the
   rewritten discovery tests. The `test_discovery.py` rewrite is the main
   work item; aim for parity with current coverage (spawn, teardown, grace
   period, idempotency) plus four new scenarios:
   - Complete array → spawn; partial topics → wait.
   - Config removal → grace-period teardown.
   - Topic disappearance → grace-period teardown.
   - Unknown topic → ignored, no spawn.
3. **End-to-end (manual):** in a kind cluster, apply
   `expected-sensors.yaml` with two arrays, start a publisher for one,
   verify Ingestor spawns only for the complete array. Edit ConfigMap to
   remove the live array; verify Ingestor torn down within ~15 s.

## Open Risks

- **ConfigMap propagation latency.** Kubelet syncs mounted ConfigMaps on
  the order of 60 s. If an operator wants faster propagation, they
  redeploy. Acceptable for now; document in README.
- **Test rewrite scope.** The discovery suite is essentially a
  ground-up rewrite, not an extension. Plan accordingly.
