# Inference Engine

Containerized inference pipeline for vehicle detection and classification using acoustic and seismic sensor data from Raspberry Shake devices.

---

## Validation status (last verified 2026-05-02)

What's been exercised end-to-end:

| Component | Status | Notes |
|-----------|--------|-------|
| Discovery → Ingestor spawn | ✅ Verified | One Ingestor per `expectedSensors` entry, per 5s poll. Discovery self-heals if its child Deployments are deleted out-of-band. |
| Ingestor → NATS JetStream | ✅ Verified | 1-second windows published as `SensorData` protobufs. Default emits raw ADC counts (mean-subtracted) to match the CRL training contract; legacy `[-1, 1]` normalization gated behind `ADC_SCALE_NORMALIZE=1`. |
| infer-detect / infer-classify | ✅ Verified | `subscribe_bind` against pre-created consumers; `TORCH_NUM_THREADS=2` pins the intra-op pool to the cgroup CPU limit (otherwise 1900ms/window thrashing). |
| Egress → ROS2 `/inference_result` | ✅ Verified | One publish per window: positives carry full `InferenceResult`, negatives carry detection-only fields. Stamped with `publish_time` + `latency_seconds` (capture-timestamp → publish, includes 1.0s window-fill floor). |
| Helm chart install | ✅ Verified | Single `helm install` brings up the pipeline. NATS bundled, KEDA optional. Note: on a kind cluster, run `scripts/build_containers.sh` (which auto-loads images into the kind node) *before* `helm install`, otherwise pods stay `Pending` with `ImagePullBackOff`. |
| Local smoke test | ✅ Verified (2026-05-02) | `scripts/local_smoke.sh` brings up the full pipeline end-to-end on a single-node kind cluster: kind + Calico + KEDA + NATS + all six inference pods. Confirmed working on Ubuntu 24.04. |
| Replay on recorded data | ✅ Independent path | `scripts/replay_in_kind.sh` brings up the cluster (no fake-publisher) on demand and injects a recording. Reuses the smoke cluster if one exists; works equally well from a clean clone. |
| Replay tool (parquet / CSV / WAV) | ✅ Verified | `scripts/replay_publisher.py` simulates a real ROS2 sensor source for testing on pre-recorded data. Format auto-detected from extension. Reports per-window precision/recall (when `present` ground-truth is in the file) and side-by-side client + cluster end-to-end latency. |
| End-to-end on synthetic data | ✅ Verified | `fake-publisher` driving the full pipeline produces `InferenceResult` ROS2 messages. |
| End-to-end on real recording | ✅ Pipeline verified | On `data_files/parsed/test/focal_audio_pickup2_rs3.parquet` + paired seismic: 60/60 windows traversed the full pipeline cleanly. Pipeline latency was within budget. |
| Model accuracy | 🔄 Re-test needed | Earlier near-random predictions traced to an ingestor-side ADC scale mismatch with the CRL training contract (training uses raw ADC counts, ingestor was dividing by `2^(bits-1)`). Fixed at `src/ingestor/buffer.py` (default off; `ADC_SCALE_NORMALIZE=1` restores legacy). After the bundle restructure landed (2026-05), the catalogs (`detect-bundles/`, `classify-bundles/`) are empty — populate them via `crl-train`'s exporter and run end-to-end against the new bundles to re-measure accuracy. |

### Required environment knobs (kind smoke test)

These are set automatically by the chart's defaults but worth knowing for
debugging:

- `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` — required when ROS2 pods share
  `hostNetwork: true` on a single node. Without this, FastDDS prefers
  shared-memory transport, but `/dev/shm` is per-pod (mount namespace ≠
  net namespace), and data is silently dropped between participants.
  Multi-host customer deployments don't need this — DDS picks UDP
  automatically when participants are on different IPs — but the value
  is safe everywhere.
- `TORCH_NUM_THREADS=2` — matches `inference.inferDetect.resources.limits.cpu`.
  PyTorch's intra-op thread pool defaults to host core count, which
  thrashes against the cgroup CPU limit. Customers who scale up CPU
  should also bump `inference.inferDetect.torchThreads` in `values.yaml`.

### Known limitations

1. **Single-namespace.** Discovery's RBAC scopes it to its own namespace.
2. **JetStream `emptyDir` by default.** `nats.storage.type: pvc` for
   production; otherwise a NATS pod restart drops in-flight detection
   results.
3. **Replay tool runs on workstation or as a one-off pod.** The
   `replay_in_kind.sh` wrapper runs it inside the kind control-plane
   container so its DDS participant lives on the same hostNetwork as
   the cluster pods. For multi-host setups the customer runs the
   script from any ROS2-enabled host on their network.

### Where to look next

- [`chart/README.md`](chart/README.md) — Helm chart values reference,
  install steps, replay tool docs.
- [`detect-bundles/README.md`](detect-bundles/README.md) and
  [`classify-bundles/README.md`](classify-bundles/README.md) — the two
  per-pod CRL bundle catalogs, ready to bake into the inference images.
  `DETECT_BUNDLE=<name>` and `CLASSIFY_BUNDLE=<name>` select bundles
  independently; defaults are the `*-default` symlinks in each catalog.
- [`scripts/install_build_host.sh`](scripts/install_build_host.sh) —
  guided Ubuntu / Debian bootstrap of the build host (docker, kubectl,
  helm, kind, poetry venv, test self-check).
- [`scripts/build_containers.sh`](scripts/build_containers.sh) — image
  build, supports `REGISTRY=… TAG=… PUSH=1` for the customer-registry
  workflow.
- [`scripts/replay_publisher.py`](scripts/replay_publisher.py) — replay
  tool source. `--help` for full CLI.
- [`scripts/install_replay.sh`](scripts/install_replay.sh) — guided
  Ubuntu / Debian setup for the replay tool's ROS2 prerequisites.
- [`scripts/tail_egress.sh`](scripts/tail_egress.sh) — one-line-per-message
  live viewer of `/inference_result`. Runs against the egress pod, no
  local ROS2 install needed.

---

## Architecture

```
ROS2 Network (Raspberry Shake sensors)
        │
        ▼
  Discovery Node  ──── k8s API ────► Ingestor Pod (per sensor array)
                                            │ JetStream: sensor.data
                                            ▼
                                     Infer Detect Node
                                      │            │
                              detected=false    detected=true
                                      │            │ JetStream: detection.result
                                      │            ▼
                                      │      Infer Classify Node
                                      │            │ JetStream: classification.result
                                      │            │
                                      ▼            ▼
                                        Egress Node
                                  (negatives via detection.result;
                                   positives via classification.result;
                                   one ROS2 publish per window)
                                             │ ROS2: /inference_result
                                             ▼
                                     InferenceResult msg
```

---

## Nodes

### Discovery (`src/discovery/`)

Reads an operator-curated **expected-sensors ConfigMap** mounted at
`/etc/inference-engine/expected-sensors.yaml`, polls the ROS2 graph every
5 s, and spawns an Ingestor Deployment for each configured array whose
required topics are all visible. Tears down an Ingestor when its array is
removed from the ConfigMap or any of its required topics goes missing for
3 consecutive polls (~15 s). Topics not listed in the ConfigMap are ignored.

The ConfigMap entry for one array names a single bundled-channel topic
the array publishes on. Discovery passes that topic into each Ingestor
via the `SENSOR_TOPIC` env var. The per-channel-tag → buffer-role
mapping lives in a separate cluster-level `ingestor-channel-map`
ConfigMap mounted into every Ingestor pod, so the customer can change
their channel-tag scheme without touching the discovery config.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMESPACE` | `default` | k8s namespace to manage deployments in |
| `TEMPLATE_PATH` | `/app/ingestor-template.yaml` | Path to the Ingestor Deployment template |
| `CONFIG_PATH` | `/etc/inference-engine/expected-sensors.yaml` | Mounted ConfigMap with the expected-pairs whitelist |

**Expected-sensors ConfigMap format** (see `k8s/expected-sensors.yaml`):

```yaml
arrays:
  shake-001:                     # array id: lowercase RFC 1123 subdomain
    audio: /shake_001/aud        # topic strings keep firmware naming
    seismic: /shake_001/ehz
    accel:                       # optional; if present, x/y/z all required
      x: /shake_001/ene
      y: /shake_001/enn
      z: /shake_001/enz
  shake-002:
    audio: /shake_002/aud
    seismic: /shake_002/ehz
```

Array IDs (the YAML keys) become Kubernetes Deployment names
(`ingestor-<array_id>`) and must therefore be **RFC 1123 subdomains**:
lowercase letters, digits, hyphens; start with a letter; end alphanumeric;
no underscores; ≤54 characters. Topic *values* can contain whatever ROS2
allows — they reflect the firmware's published topic names.

Edits to the ConfigMap propagate via kubelet sync (~60 s) and are picked up
on the next 5 s poll without restarting Discovery.

---

### Ingestor (`src/ingestor/`)

Subscribes to one ROS2 topic per role configured by Discovery. Each
subscription's callback is bound at construction time to its buffer role
(`acoustic`, `seismic`, `accel_x|y|z`), so the Ingestor no longer parses
`msg.sensor_id` to recover the role. Buffers 1-second windows, applies
per-window mean subtraction (DC removal), and publishes `SensorData`
protobufs to NATS.

**Default scaling (matches the CRL training contract):** raw ADC integer
counts are cast to float32 and only DC-corrected — bit-depth scale is
preserved. This matches what `crl-train` saw at training time (parquet
`amplitude` column stored raw counts; only `remove_dc` was applied).

**Legacy `[-1, 1]` mode** is available for the older
`WaveformClassificationCNN` model by setting `ADC_SCALE_NORMALIZE=1`,
which divides by `2^(bits-1)` before mean subtraction:

| Channel | Bit depth | Scale (legacy mode only) |
|---------|-----------|--------------------------|
| acoustic | 16-bit | `2^15` |
| seismic / accel | 24-bit | `2^23` |

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | — | NATS server address (e.g. `nats://nats-service:4222`) |
| `SENSOR_ARRAY` | — | Sensor array identifier (e.g. `shake_001`) |
| `SENSOR_TOPIC` | — | Bundled-channel ROS2 topic the array publishes on (injected by Discovery) |
| `CHANNEL_MAP_PATH` | `/etc/inference-engine/channels.yaml` | Path to the channel-tag → role YAML; mounted from the `ingestor-channel-map` ConfigMap |
| `NATIVE_RATES` | `0` | When `1`, ship each channel at its native rate (CRL expects audio=16k, seismic=100). When `0`, upsample everything to `TARGET_RATE` for the legacy CNN path. |
| `ADC_SCALE_NORMALIZE` | `0` | When `1`, divide each channel by `2^(bits-1)` before mean subtraction (legacy `[-1, 1]` contract). When `0`, preserve raw counts. |
| `AUDIO_BIT_DEPTH` / `SEISMIC_BIT_DEPTH` / `ACCEL_BIT_DEPTH` | `16` / `24` / `24` | Used only when `ADC_SCALE_NORMALIZE=1`. Adjust if hardware bit depth differs. |

**Buffer roles:** `acoustic`, `seismic`, `accel_x`, `accel_y`, `accel_z`.
`acoustic` and `seismic` are required; the three `accel_*` roles must
appear together or not at all. The customer's free-form channel tags
(e.g. `MIC`, `EHZ`) map to these roles via `channels.yaml`.

---

### Infer Detect (`src/infer_detect/`)

Subscribes to `sensor.data` NATS subject. Runs the vehicle detection model and publishes `DetectionResult` to `detection.result`. Pending model integration.

---

### Infer Classify (`src/infer_classify/`)

Subscribes to `detection.result` NATS subject. Runs the vehicle classification model and publishes `EgressPayload` to `classification.result`. Only processes windows where `vehicle_detected == true`. Pending model integration.

---

### Egress (`src/egress/`)

Subscribes to JetStream's `detection.result` and `classification.result`
streams (queue group `egress`) and publishes `InferenceResult` ROS2
messages on `/inference_result`. **One message per inference window:**
positives flow through the classifier and arrive on `classification.result`;
negatives surface directly from `detection.result`. The earlier double-
publish behavior is gone.

Each `InferenceResult` carries two latency-instrumentation fields,
populated for both positives and negatives:

| Field | Type | Meaning |
|-------|------|---------|
| `publish_time` | `float64` (Unix epoch s) | Wall-clock moment egress published this result. Same clock source as the upstream message's `timestamp_unix`. |
| `latency_seconds` | `float64` | `publish_time` − the window's original capture timestamp. **Includes the 1.0 s window-fill duration as a floor** — a value of `~1.05` means ~50 ms of pipeline compute, not "broken." |

These flow through to ROS2 subscribers — `replay_publisher.py` reports
them in its summary block, and `scripts/tail_egress.sh` prints them per
message. NTP sync between the sensor pod and egress pod is required for
the latency to be meaningful (already an assumption baked into the
existing pipeline).

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `NATS_URL` | NATS server address |

---

## Message Flow

| NATS Subject | Protobuf type | Published by | Consumed by |
|-------------|---------------|-------------|-------------|
| `sensor.data` | `SensorData` | Ingestor | Infer Detect |
| `detection.result` | `DetectionResult` | Infer Detect | Infer Classify, Egress |
| `classification.result` | `EgressPayload` | Infer Classify | Egress |

---

## ROS2 Messages

| Message | Package | Used by |
|---------|---------|---------|
| `std_msgs/String` (JSON payload) | `std_msgs` | Ingestor (subscriber) |
| `InferenceResult` | `ros2_interfaces` | Egress (publisher) |

The bundled-channel JSON payload carries `sensor_id`, `state`,
`timestamp_unix`, and a `channels` list with `{channel, sampling_rate,
dt, readings}` per entry. See `src/ingestor/dispatch.py` for the
parser and the channel-tag → buffer-role mapping.

**If you change a `.msg` file**, every host that runs the replay tool
or otherwise links against `ros2_interfaces` outside the cluster must
rebuild its colcon workspace, since the package emits compiled C++
type-support libraries that are platform-specific. The cluster pods
get a fresh build automatically when you re-run
`scripts/build_containers.sh`. For replay hosts, re-run
`scripts/install_replay.sh` (or just the colcon step inside
`~/ros2_replay_ws`). Without this, subscribers will silently fail to
deserialize the new fields.

---

## Shared Protobuf Package

All inter-node protobuf definitions live in `inference-protos/`. Each node installs this package via pip in its Dockerfile — it is not a PyPI package.

**To recompile after editing `inference.proto`:**

```bash
cd inference-engine
poetry install
poetry run python scripts/compile_protos.py
```

---

## Testing

**Test suite status (2026-05-02): ✅ 167 passed, 3 skipped (rclpy-only), 0 failed.**

The inference engine has comprehensive test coverage across all nodes.
`tests/conftest.py` sets `NATIVE_RATES=1` by default so per-channel
buffer assertions work without manual env setup; tests opt into legacy
behavior by `monkeypatch`-ing `ingestor.buffer._ADC_SCALE_NORMALIZE`.

**Setup (one time):**

Test dependencies live in the `test` group in `pyproject.toml`. `torch`
and `torchaudio` are installed separately from the CPU-only PyTorch
wheel index, mirroring `src/ingestor/Dockerfile:18-24`. The default
PyPI wheels pull in CUDA shared libraries (`libcudart.so`) that fail
to dlopen on hosts without an NVIDIA driver, so we sidestep them. CPU
wheels do not affect the deployed model — production containers
install their own torch at build time.

Run these four commands from the `inference-engine/` directory:

```bash
poetry install --with test
poetry run pip install ./inference-protos
poetry run pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio
poetry run pytest tests/ -v
```

Expected outcome on Linux x86_64 (verified 2026-05-02):
**167 passed, 3 skipped (rclpy-only), 0 failed.**

**Run subsets:**

```bash
# By node
poetry run pytest tests/test_discovery.py -v       # Discovery
poetry run pytest tests/test_ingestor.py -v        # Ingestor (dispatch / role map)
poetry run pytest tests/test_buffer.py -v          # SensorBuffer (window math, both ADC modes)
poetry run pytest tests/test_egress.py -v          # Egress (protobuf, NATS, ROS2)
poetry run pytest tests/test_infer_detect.py -v    # Vehicle Detection
poetry run pytest tests/test_infer_classify.py -v  # Vehicle Classification

# By category
poetry run pytest -m unit -v         # Unit tests only
poetry run pytest -m integration -v  # Integration tests only
poetry run pytest -m asyncio -v      # Async tests only
```

**Troubleshooting:**

- **`pytest: command not found`** after `poetry install --with test`:
  you're invoking pytest from outside the poetry venv. Either run
  `poetry shell` first or prefix every command with `poetry run` (as
  shown above). The system suggestion to
  `sudo apt install python3-pytest` would install pytest globally and
  bypass poetry's pinned dev deps — don't follow it.

- **`OSError: libcudart.so.13: cannot open shared object file`** on
  `import torchaudio`: you have the CUDA-enabled wheel from PyPI and
  need to reinstall the CPU build:
  ```bash
  poetry run pip uninstall -y torch torchaudio
  poetry run pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio
  ```

- **`ModuleNotFoundError: No module named 'inference_protos'`** during
  test collection: the `poetry run pip install ./inference-protos`
  step in the setup didn't complete. Re-run it.

**Test coverage:**
- **Discovery Node**: ConfigMap parsing, completeness checks, PollState grace logic, manifest construction
- **Ingestor Node**: channel-map loading, JSON-message dispatch, single subscription wiring, window-close rate validation
- **SensorBuffer**: window math, holding-pen logic, packaging contract — both default (raw counts) and legacy (`ADC_SCALE_NORMALIZE=1`) modes
- **Egress Node**: protobuf conversion, NATS subscriptions, ROS2 publishing (including new `publish_time` / `latency_seconds` fields), edge cases
- **Infer Detect Node**: model loading (CUDA/MPS/CPU), binary detection, tensor preprocessing, NATS integration
- **Infer Classify Node**: multi-class classification, type-head wiring, CLASS_MAP, confidence scoring

The 3 skipped tests in `test_egress.py` require a working `rclpy`
import; they run in CI when ROS2 is on the path. See
[tests/README.md](tests/README.md) for detailed test documentation.

---

## Build and Deploy

### Prerequisites (from a fresh clone)

On the build host (a Linux box with Docker; Ubuntu 24.04 is the
canonical target). For Ubuntu / Debian, run the bootstrap script and
skip the rest of this section:

```bash
cd inference-engine
bash scripts/install_build_host.sh
```

It checks for `docker`, `kubectl`, `helm`, `kind`, and `poetry`,
prompts before installing each missing piece, sets up the poetry venv
with the test group + CPU-only torch, installs `inference-protos`, and
runs the test suite as a self-check. Pass `ASSUME_YES=1` to skip the
prompts. KEDA stays optional and isn't installed.

For other platforms — or if you'd rather check things by hand — these
are what the script verifies:

1. **Tooling on PATH**: `docker`, `kubectl`, `helm`, `kind`, `poetry`.
   KEDA is *optional* — install it only if you want autoscaling on
   JetStream backlog depth.
2. **CRL inference bundles** under `detect-bundles/` and
   `classify-bundles/`. Each pod is selected independently —
   `DETECT_BUNDLE=<name>` for `infer-detect` (default `detect-default`)
   and `CLASSIFY_BUNDLE=<name>` for `infer-classify` (default
   `classify-default`). See [detect-bundles/README.md](detect-bundles/README.md)
   and [classify-bundles/README.md](classify-bundles/README.md) for the
   per-pod catalogs and selection rules.
3. **`kubectl` context** pointing at the destination cluster.

`crl-train` is **not** a build-time dependency for customers. The
training pipeline produces TorchScript bundles once on a dev box and
commits them under the per-pod catalogs; from there the inference
build just copies the artifacts. The crl-train fallback path in
`scripts/build_containers.sh` is for dev re-exports — you can ignore
it.

`compile_protos.py` is **not** required for a fresh clone — the
generated `inference_pb2.py` is committed and is pure Python (arch
independent). Re-run it only if you've edited `protos/inference.proto`.

Two deployment paths are supported:

- **Helm chart** (`chart/`) — the customer-facing path. Single
  `helm install`, parameterized via `values.yaml`. See
  [`chart/README.md`](chart/README.md) for full documentation.
- **Raw manifests** (`k8s/`) — used by `scripts/local_smoke.sh` for
  development. Each manifest can be `kubectl apply`'d individually.

### Helm install (recommended for real clusters)

For a real Kubernetes cluster (not kind):

```bash
# 1. Build, tag, and push containers to your registry
REGISTRY=registry.example.com/dark-circle TAG=v0.1.0 PUSH=1 \
    scripts/build_containers.sh

# 2. Configure values for your site
cp chart/values.yaml my-site-values.yaml
# Edit images.registry, images.tag, expectedSensors at minimum.
# expectedSensors schema is the same as the Discovery ConfigMap shown
# in the "Discovery" section above.

# 3. Install
helm install dark-circle ./chart -f my-site-values.yaml
```

> **For local kind testing, use `scripts/local_smoke.sh` instead** — it
> handles cluster creation, Calico CNI, image build + load, manifest
> apply, and pod readiness in one command. The Helm path on kind needs
> `scripts/build_containers.sh` to run **before** `helm install` so
> images are loaded into the kind node; otherwise pods stay `Pending`.

For multi-host deployments where ROS2 participants live on different
machines, also configure `ros2.fastddsProfile` in `values.yaml` to
declare explicit unicast peers. Single-host kind setups don't need
this — see [kind cluster DDS configuration](#kind-cluster-dds-configuration)
for the reasoning.

Ingestor pods are spawned automatically by the Discovery node once an
array listed in `expectedSensors` has all of its required topics visible
on the ROS2 graph. infer-detect / infer-classify / egress run with fixed
`replicas: 1` by default; set `keda.enabled: true` in values.yaml for
backlog-driven autoscaling.

### Verifying the install

```bash
# 1. Discovery should see the cluster and start polling.
kubectl logs -n default -l app=discovery -f
# Within ~10s of your sensors publishing, expect: "Spawned ingestor for <array-id>"

# 2. Confirm the per-array Ingestor pods came up.
kubectl get pods -n default

# 3. Tail inference results in real time. Prints one compact line per
#    InferenceResult — sensor id, capture timestamp, presence/class, and
#    the cluster-stamped latency_seconds.
scripts/tail_egress.sh

# 4. (Optional) Verify NATS JetStream consumers are draining.
kubectl exec -n default deploy/nats -- \
    nats --server=localhost:4222 consumer info SENSOR_DATA infer-detect
```

A healthy `latency_seconds` value sits just above 1.0 — the 1.0 s
window-fill duration is a floor, not a bug. Values much higher than
that indicate queue depth or compute pressure (see the egress
section). The full `ros2 topic echo` output is also available via
`kubectl exec deploy/egress -- /bin/bash -c '... ros2 topic echo
/inference_result'` if you want raw YAML.

### Raw manifest install (development)

```bash
# 1. Build containers (loads into the dark-circle kind cluster if present)
bash scripts/build_containers.sh

# 2. Deploy NATS + streams + consumers
kubectl apply -f k8s/nats/nats-deployment.yaml
kubectl apply -f k8s/nats/nats-service.yaml
kubectl apply -f k8s/nats/jetstream-streams.yaml

# 3. Deploy ConfigMaps, RBAC, Discovery
kubectl apply -f k8s/inference-engine-config.yaml
kubectl apply -f k8s/sensor-config.yaml
kubectl apply -f k8s/expected-sensors.yaml
kubectl apply -f k8s/rbac/
kubectl apply -f k8s/discovery.yaml

# 4. Deploy inference + egress
kubectl apply -f k8s/infer-detect.yaml
kubectl apply -f k8s/infer-classify.yaml
kubectl apply -f k8s/egress.yaml

# 5. Optional: KEDA autoscaling (requires KEDA pre-installed)
kubectl apply -f k8s/keda/
```

### NATS JetStream streams

The pipeline uses three streams, each retaining one minute of messages
(real-time data, no replay value beyond a brief crash recovery window):

| Stream                  | Subject                  | Producer       | Consumer (durable, queue) |
|-------------------------|--------------------------|----------------|---------------------------|
| `SENSOR_DATA`           | `sensor.data`            | Ingestor       | `infer-detect`            |
| `DETECTION_RESULT`      | `detection.result`       | Infer Detect   | `infer-classify`, `egress-detection` |
| `CLASSIFICATION_RESULT` | `classification.result`  | Infer Classify | `egress-classification`   |

Queue groups distribute messages across replicas of the same consumer,
so scaling infer-detect from 1 to 3 splits work three ways instead of
delivering each window three times.

---

## Three customer modes

Bringing up the pipeline on a single Linux server has three distinct
modes, each independent of the others. They share a kind cluster
created on demand by whichever script you run first.

| Mode | Entry point | What it does | When to use |
|------|------------|--------------|-------------|
| **Smoke** | `bash scripts/local_smoke.sh` | Cluster + synthetic fake-publisher driving every topic | One-time sanity check that the build + cluster-up logic works on this host. |
| **Replay** | `bash scripts/replay_in_kind.sh <audio> <seismic>` | Cluster (no fake-publisher) + a one-off pod publishing recorded data | Real work on pre-recorded recordings. |
| **Live** | point real ROS2 sensors at the cluster, no script | Cluster (no fake-publisher) + real Raspberry Shake topics | Real work on live sensor input. |

All three modes share the same cluster; you can switch between them
without re-running the build or recreating the cluster. The
fake-publisher Deployment is automatically removed when you switch
from Smoke into Replay or Live, so synthetic data doesn't compete
with real input on the same topics.

### Smoke test (one-time sanity check)

```bash
bash scripts/local_smoke.sh
```

Creates a kind cluster named `dark-circle` (Calico CNI, pod CIDR
`192.168.0.0/16`), installs KEDA, builds the six images, loads them
into the kind node, applies all manifests including
`fake-publisher.yaml`, and waits for every Deployment to be Ready.

End-to-end runtime: ~5–10 minutes the first time (image pulls and
torch wheel downloads), ~30 seconds on rebuild.

Tear down when finished:

```bash
bash scripts/local_smoke.sh --teardown
```

After the first successful smoke, you typically don't run this again.
Replay and Live use the same cluster but skip fake-publisher.

### Replay on recorded data

```bash
bash scripts/replay_in_kind.sh <audio.parquet|csv|wav> <seismic.parquet|csv|wav> [replay flags...]
```

Brings up the cluster automatically if it isn't already running
(without fake-publisher), then injects the recording as a one-off
hostNetwork-joined pod publishing on the same ROS2 topics the
pipeline expects. The pipeline can't tell this apart from a real
Raspberry Shake.

**File format** is auto-detected from the extension:

| Format | Sample rate source | Ground truth | Notes |
|--------|-------------------|--------------|-------|
| `.parquet` | `time_stamp` column (or `--audio-rate` / `--seismic-rate`) | `present` column if present | Default schema; preserves int amplitude. |
| `.csv` | same as parquet | same | Header row required unless `--no-header`. |
| `.wav` | WAV header | none — no per-window scoring | Mono PCM only (8 / 16 / 24 / 32-bit). Useful when you have a raw acoustic capture without ground-truth labels. |

Replay subscribes back to `/inference_result` and prints per-window
predictions and a summary block at the end (precision/recall against
ground-truth `present` when available, plus client + cluster latency
percentiles).

**Common replay flags** (forwarded through to `replay_publisher.py`;
see `replay_publisher.py --help` for the full list):

| Flag | Default | Effect |
|------|---------|--------|
| `--start-second <S>` | `0` | Start replay `S` seconds into the recording. Validated against both files' lengths (errors out if negative or past the shorter recording) and snapped to the nearest 0.1s tick so audio + seismic stay aligned. |
| `--duration <S>` | (full file) | Cap replay length to `S` seconds *from `--start-second`*. |
| `--no-subscribe` | (subscribe) | Skip subscribing to `/inference_result` — useful when you want pure data injection without the latency/scoring overlay. |

Examples:

```bash
# Play the full recording from the beginning (default)
bash scripts/replay_in_kind.sh audio.parquet seismic.parquet

# Play 30 seconds starting at 1:00 into the recording
bash scripts/replay_in_kind.sh audio.parquet seismic.parquet \
    --start-second 60 --duration 30

# Cluster already up? skip the bring-up check
SKIP_CLUSTER_UP=1 bash scripts/replay_in_kind.sh audio.parquet seismic.parquet
```

### Live inference on real sensors

No script needed. With the cluster up (from a prior smoke or replay
invocation), point real Raspberry Shake topics at it:

1. Edit `k8s/expected-sensors.yaml` (or your Helm values' `expectedSensors`)
   to list your real array IDs and topics.
2. `kubectl apply -f k8s/expected-sensors.yaml` if you used the
   smoke/replay path; `helm upgrade` if Helm.
3. Discovery picks the change up on its next 5-second poll and spawns
   an Ingestor for each visible array.
4. Watch results: `bash scripts/tail_egress.sh`.

If you've been running smoke or replay and want to switch to live, no
explicit teardown is needed — Discovery already only attaches to
arrays listed in `expected-sensors.yaml`. Just remove `fake-publisher`
(`kubectl delete deployment fake-publisher`) and update the ConfigMap.

### Re-running after a failed bring-up

If a smoke or replay run was interrupted partway through (CNI
half-installed, control plane wedged, etc.), tear down before retrying
— the cluster's "already exists, reusing" branch will otherwise skip
re-installing whatever broke:

```bash
bash scripts/local_smoke.sh --teardown
bash scripts/local_smoke.sh   # or replay_in_kind.sh
```

### kind cluster DDS configuration

Several DDS settings are required when all ROS2 pods share the kind
control-plane's hostNetwork. They're set automatically by the chart and
the raw manifests, but worth understanding:

1. **`hostNetwork: true`** on Discovery / Ingestor / Egress / fake-publisher.
   Calico-VXLAN doesn't carry multicast, so without hostNetwork these
   pods can't discover each other's ROS2 graph.
2. **`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`** in the
   `inference-engine-config` ConfigMap. With multiple participants on
   the same IP, FastDDS prefers shared-memory transport — but
   `/dev/shm` is per-pod (mount namespace ≠ net namespace), so
   publishes go into one pod's `/dev/shm` and reads come from
   another's, producing silent data loss. Forcing UDPv4 sidesteps it.

Real multi-host customer deployments need neither workaround — DDS
picks UDP automatically when participants are on different IPs, and
multicast is the customer's network admin's problem (or use the
`ros2.fastddsProfile` knob in `values.yaml` for explicit unicast peers).

### CRL inference deployment

The pipeline deploys two independent CRL bundles — one per inference pod:

- **`infer-detect`** carries the (encoder + presence head) for vehicle
  detection. The bundle is selected at build time by `DETECT_BUNDLE`
  (default `detect-default`) and resolved against
  [`inference-engine/detect-bundles/`](detect-bundles/README.md).
- **`infer-classify`** carries the (encoder + type head) for vehicle
  classification. Selected by `CLASSIFY_BUNDLE` (default
  `classify-default`) against
  [`inference-engine/classify-bundles/`](classify-bundles/README.md).

The two pods are independent: each carries its own copy of the encoder
it pairs with, and the classify pod re-encodes the inbound waveform
rather than depending on the detect pod's latent.

Each inference image carries a self-contained TorchScript bundle —
a small directory of `.ts` files plus a `meta.json`. Customers don't
need `crl-train` installed; they just pick a bundle (or accept the
default `*-default` symlinks).

`scripts/build_containers.sh` reads `DETECT_BUNDLE` / `CLASSIFY_BUNDLE`
and resolves them against the per-pod catalogs.

Override the bundles for a customer-specific build:

```bash
DETECT_BUNDLE=<detect-bundle-name> CLASSIFY_BUNDLE=<classify-bundle-name> \
    bash scripts/build_containers.sh infer-detect infer-classify

# Roll the inference pods (Helm install)
kubectl rollout restart deploy/infer-detect deploy/infer-classify

# Or for raw-manifest install
kubectl rollout restart deploy/infer-detect deploy/infer-classify -n default
```

The capstone targets are `pres_f1 ≥ 0.85` and `type_f1 ≥ 0.70`. The
selection rules in each catalog README enforce floors below those
targets so a regression isn't auto-promoted into the shipping default.

The wire protocol carries no latent — `DetectionResult` has only
`sensor_data + vehicle_detected + confidence`. Classify re-encodes
from `sensor_data.acoustic_data` / `seismic_data` using its own
bundle's window sizes.

Per-sensor mode in `infer-detect` combines the two presence heads via
OR; in `infer-classify` the two type heads are averaged before argmax.

#### Producing a new bundle (dev only)

When a new training run beats the current leader, re-export and commit
the new bundle. This requires `crl-train` installed alongside
`inference-engine`. The exporter writes directly into the appropriate
per-kind catalog when given `--bundle-name`:

```bash
cd crl-train

# Detect bundle (encoder + presence head)
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind detect \
    --bundle-name <frontend>-<mode>-<run>-v<N>

# Classify bundle (encoder + type head)
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind classify \
    --bundle-name <frontend>-<mode>-<run>-<probe>-v<N>
```

To evaluate a bundle against the catalog and repoint its `*-default`
symlink if it wins, add `--promote-default`:

```bash
poetry run python export_for_inference.py \
    --save-dir <save-dir> \
    --bundle-kind classify \
    --bundle-name <name> \
    --promote-default
```

After exporting, commit the new bundle directory and update the
catalog table in the relevant README. See
[`crl-train/README.md`](../crl-train/README.md) for the full deployment
workflow.

---

## Hardware

Designed for **Raspberry Shake** devices (RS1D and RS4D) with an external 16-bit USB/I2S microphone for acoustic data. Deployed on Ubuntu 24.04 / `x86_64`. Developed on macOS.
