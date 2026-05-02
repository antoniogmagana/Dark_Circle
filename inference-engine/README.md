# Inference Engine

Containerized inference pipeline for vehicle detection and classification using acoustic and seismic sensor data from Raspberry Shake devices.

---

## Validation status (last verified 2026-05-01)

What's been exercised end-to-end:

| Component | Status | Notes |
|-----------|--------|-------|
| Discovery → Ingestor spawn | ✅ Verified | One Ingestor per `expectedSensors` entry, per 5s poll. Discovery self-heals if its child Deployments are deleted out-of-band. |
| Ingestor → NATS JetStream | ✅ Verified | 1-second windows published as `SensorData` protobufs. |
| infer-detect / infer-classify | ✅ Verified | `subscribe_bind` against pre-created consumers; `TORCH_NUM_THREADS=2` pins the intra-op pool to the cgroup CPU limit (otherwise 1900ms/window thrashing). |
| Egress → ROS2 `/inference_result` | ✅ Verified | One publish per window: positives carry full `InferenceResult`, negatives carry detection-only fields. |
| Helm chart install | ✅ Verified | Single `helm install` brings up the pipeline. NATS bundled, KEDA optional. |
| Replay tool (parquet + CSV) | ✅ Verified | `scripts/replay_publisher.py` simulates a real ROS2 sensor source for testing on pre-recorded data; reports per-window precision/recall + end-to-end latency. |
| End-to-end on synthetic data | ✅ Verified | `fake-publisher` driving the full pipeline produces `InferenceResult` ROS2 messages. |
| End-to-end on real recording | ✅ Pipeline verified | On `data_files/parsed/test/focal_audio_pickup2_rs3.parquet` + paired seismic: 60/60 windows traversed the full pipeline cleanly. Pipeline latency was within budget. |
| Model accuracy | ⚠️ Out of scope here | The CRL run baked into images at `crl-train/saved_crl/runs/multiscale/vae/v1` is from an older 3-branch frontend architecture. Predictions on real test recordings hover at threshold (~0.5 detection conf, classifier near-uniform across classes). Re-export from a current 4-branch checkpoint is required before model results are meaningful. This is a model-side task, not a cluster bug. |

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
- [`scripts/replay_publisher.py`](scripts/replay_publisher.py) — replay
  tool source. `--help` for full CLI.
- [`scripts/install_replay.sh`](scripts/install_replay.sh) — guided
  Ubuntu / Debian setup for the replay tool's ROS2 prerequisites.
- [`scripts/build_containers.sh`](scripts/build_containers.sh) — image
  build, supports `REGISTRY=… TAG=… PUSH=1` for the customer-registry
  workflow.

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

The ConfigMap entry for one array names the audio and seismic topics
explicitly; an optional `accel` block names the x/y/z accelerometer topics.
Discovery passes those role bindings into each Ingestor via the
`SENSOR_ROLE_MAP` env var, so role assignment is configuration, not a topic-
naming convention.

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
`msg.sensor_id` to recover the role. Buffers 1-second windows, normalizes
to `[-1, 1]` using ADC full-scale, and publishes `SensorData` protobufs
to NATS.

**ADC normalization:**

| Channel | Bit depth | Scale |
|---------|-----------|-------|
| acoustic | 16-bit | `2^15` |
| seismic / accel | 24-bit | `2^23` |

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `NATS_URL` | NATS server address (e.g. `nats://nats-service:4222`) |
| `SENSOR_ARRAY` | Sensor array identifier (e.g. `shake_001`) |
| `SENSOR_ROLE_MAP` | JSON object `{role: topic}` injected by Discovery, e.g. `{"acoustic":"/shake_001/aud","seismic":"/shake_001/ehz"}` |

**Buffer roles:** `acoustic`, `seismic`, `accel_x`, `accel_y`, `accel_z`.
`acoustic` and `seismic` are required; the three `accel_*` roles must
appear together or not at all.

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
| `RawSensorReading` | `ros2_interfaces` | Ingestor (subscriber) |
| `InferenceResult` | `ros2_interfaces` | Egress (publisher) |

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

**Test Suite Status: ✅ 125/125 tests passing (100% complete)**

The inference engine has comprehensive test coverage across all nodes:

```bash
# Run all tests (excluding CUDA-dependent test_buffer.py)
pytest tests/ --ignore=tests/test_buffer.py -v

# Run integration and inference tests
pytest tests/test_discovery.py tests/test_ingestor.py tests/test_egress.py \
       tests/test_infer_detect.py tests/test_infer_classify.py -v

# Run by node
pytest tests/test_discovery.py -v       # 37 tests (Discovery)
pytest tests/test_ingestor.py -v        # 21 tests (Ingestor)
pytest tests/test_egress.py -v          # 21 tests (Egress)
pytest tests/test_infer_detect.py -v    # 23 tests (Vehicle Detection)
pytest tests/test_infer_classify.py -v  # 23 tests (Vehicle Classification)

# Run by category
pytest -m unit -v         # Unit tests only
pytest -m integration -v  # Integration tests only
pytest -m asyncio -v      # Async tests only
```

**Test Coverage:**
- **Discovery Node** (37 tests): ConfigMap parsing, completeness checks, PollState grace logic, manifest construction
- **Ingestor Node** (21 tests): SENSOR_ROLE_MAP parsing, role-bound callbacks, subscription wiring, ADC normalization
- **Egress Node** (21 tests): Protobuf conversion, NATS subscriptions, ROS2 publishing, edge cases
- **Infer Detect Node** (23 tests): Model loading (CUDA/MPS/CPU), binary detection, tensor preprocessing, NATS integration
- **Infer Classify Node** (23 tests): Multi-class classification, Mel spectrograms, CLASS_MAP, confidence scoring
- **SensorBuffer** (33 tests): Comprehensive buffering logic in `test_buffer.py` (requires CUDA)

See [tests/README.md](tests/README.md) for detailed test documentation.

---

## Build and Deploy

**Prerequisites:** Docker, kubectl, and a Kubernetes cluster.
KEDA is *optional* (off by default); enable it only if you need to
autoscale the inference + egress pods on JetStream backlog depth.

Two deployment paths are supported:

- **Helm chart** (`chart/`) — the customer-facing path. Single
  `helm install`, parameterized via `values.yaml`. See
  [`chart/README.md`](chart/README.md) for full documentation.
- **Raw manifests** (`k8s/`) — used by `scripts/local_smoke.sh` for
  development. Each manifest can be `kubectl apply`'d individually.

### Helm install (recommended)

```bash
# 1. Compile protobufs
poetry run python scripts/compile_protos.py

# 2. Build, tag, and push containers to your registry
REGISTRY=registry.example.com/dark-circle TAG=v0.1.0 PUSH=1 \
    scripts/build_containers.sh

# 3. Configure values for your site
cp chart/values.yaml my-site-values.yaml
# Edit images.registry, images.tag, expectedSensors at minimum.

# 4. Install
helm install dark-circle ./chart -f my-site-values.yaml
```

Ingestor pods are spawned automatically by the Discovery node once an
array listed in `expectedSensors` has all of its required topics visible
on the ROS2 graph. infer-detect / infer-classify / egress run with fixed
`replicas: 1` by default; set `keda.enabled: true` in values.yaml for
backlog-driven autoscaling.

### Raw manifest install (development)

```bash
# 1. Compile protobufs
poetry run python scripts/compile_protos.py

# 2. Build containers (loads into the dark-circle kind cluster if present)
bash scripts/build_containers.sh

# 3. Deploy NATS + streams + consumers
kubectl apply -f k8s/nats/nats-deployment.yaml
kubectl apply -f k8s/nats/nats-service.yaml
kubectl apply -f k8s/nats/jetstream-streams.yaml

# 4. Deploy ConfigMaps, RBAC, Discovery
kubectl apply -f k8s/inference-engine-config.yaml
kubectl apply -f k8s/sensor-config.yaml
kubectl apply -f k8s/expected-sensors.yaml
kubectl apply -f k8s/rbac/
kubectl apply -f k8s/discovery.yaml

# 5. Deploy inference + egress
kubectl apply -f k8s/infer-detect.yaml
kubectl apply -f k8s/infer-classify.yaml
kubectl apply -f k8s/egress.yaml

# 6. Optional: KEDA autoscaling (requires KEDA pre-installed)
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

## Local smoke test (Linux kind)

A scripted end-to-end verification on a single-node kind cluster, with
KEDA + JetStream + the full inference pipeline. The fake publisher inside
the cluster drives the ROS2 graph so no real Raspberry Shake hardware is
required.

**Prerequisites** (one-time):
```bash
# Linux. Docker, kubectl, kind, and helm must be on PATH.
sudo apt install docker.io kubectl
go install sigs.k8s.io/kind@latest          # or your distro's package
curl -L https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

**Run the smoke test:**
```bash
cd inference-engine
scripts/local_smoke.sh
```

The script creates a kind cluster named `dark-circle` (with kindnet
disabled and Calico installed for multicast-capable pod networking),
installs KEDA, builds the six images, applies all manifests, and prints
the next steps for inspecting JetStream consumers and the ROS2 output.

**Tear down** the cluster:
```bash
scripts/local_smoke.sh --teardown
```

The fake publisher emits `/shake_001/aud` and `/shake_001/ehz` (and the
optional accel topics) at 10 Hz with synthetic samples — enough for
Discovery to recognize the array as complete and for the Ingestor to fill
its buffer.

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

The pipeline deploys the CRL frontend + presence head in `infer-detect`
and the type head in `infer-classify`. The CRL TorchScript bundle is
baked into the inference images at build time from a saved CRL run
directory. The default run dir in `scripts/build_containers.sh` is
`crl-train/saved_crl/runs/multiscale/vae/v1` — **as of 2026-05-01 this
is a stale 3-branch-frontend checkpoint that does not load against
crl-train's current 4-branch architecture and produces near-random
predictions on real test recordings.** Pick a current keeper run when
preparing customer images:

```bash
# Re-export and rebuild for a specific saved run
CRL_RUN_DIR=crl-train/saved_crl/runs/<RUN>/<VARIANT>/<PHASE> \
    CRL_FORCE_EXPORT=1 \
    scripts/build_containers.sh infer-detect infer-classify

# Roll the inference pods (Helm install)
kubectl rollout restart deploy/infer-detect deploy/infer-classify

# Or for raw-manifest install
kubectl rollout restart deploy/infer-detect deploy/infer-classify -n default
```

The wire protocol (`DetectionResult.z_fused` vs `z_audio + z_seismic`)
and the inference pods auto-detect mode from `meta.json` baked into the
image. Per-sensor mode combines the two presence heads via OR; the type
heads are averaged before argmax.

---

## Hardware

Designed for **Raspberry Shake** devices (RS1D and RS4D) with an external 16-bit USB/I2S microphone for acoustic data. Deployed on Ubuntu 24.04 / `x86_64`. Developed on macOS.
