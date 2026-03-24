# Inference Engine

Containerized inference pipeline for vehicle detection and classification using acoustic and seismic sensor data from Raspberry Shake devices.

---

## Architecture

```
ROS2 Network (Raspberry Shake sensors)
        │
        ▼
  Discovery Node  ──── k8s API ────► Ingestor Pod (per sensor array)
                                            │ NATS: sensor.data
                                            ▼
                                     Infer Detect Node
                                      │            │
                              detected=false    detected=true
                                      │            │ NATS: detection.result
                                      │            ▼
                                      │      Infer Classify Node
                                      │            │ NATS: classification.result
                                      │            │
                                      └──────┬─────┘
                                             ▼
                                        Egress Node
                                             │ ROS2: /inference_result
                                             ▼
                                     InferenceResult msg
```

---

## Nodes

### Discovery (`src/discovery/`)

Watches the ROS2 network for active `RawSensorReading` topics. Groups topics by sensor array prefix and dynamically spawns or tears down Ingestor deployments via the k8s API.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NAMESPACE` | `default` | k8s namespace to manage deployments in |
| `TEMPLATE_PATH` | `/app/ingestor-template.yaml` | Path to the Ingestor Deployment template |

---

### Ingestor (`src/ingestor/`)

Subscribes to all ROS2 topics for a single sensor array. Buffers 1-second windows of sensor data, normalizes to `[-1, 1]` using ADC full-scale, and publishes `SensorData` protobufs to NATS.

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
| `SENSOR_TOPICS` | Comma-separated list of ROS2 topics to subscribe to |

**Channel codes:**

| ROS2 code | Buffer channel |
|-----------|---------------|
| `aud` | acoustic |
| `ehz` | seismic |
| `ene` | accel_x |
| `enn` | accel_y |
| `enz` | accel_z |

---

### Infer Detect (`src/infer_detect/`)

Subscribes to `sensor.data` NATS subject. Runs the vehicle detection model and publishes `DetectionResult` to `detection.result`. Pending model integration.

---

### Infer Classify (`src/infer_classify/`)

Subscribes to `detection.result` NATS subject. Runs the vehicle classification model and publishes `EgressPayload` to `classification.result`. Only processes windows where `vehicle_detected == true`. Pending model integration.

---

### Egress (`src/egress/`)

Subscribes to `detection.result` and `classification.result` NATS subjects. Publishes `InferenceResult` ROS2 messages to `/inference_result`. Publishes immediately on detection (if positive) and again on classification.

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `NATS_URL` | NATS server address |
| `SENSOR_ARRAY` | Sensor array identifier |

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

## Build and Deploy

**Prerequisites:** Docker, kubectl configured against your cluster.

```bash
# 1. Compile protobufs
poetry run python scripts/compile_protos.py

# 2. Build and push containers
bash scripts/build_containers.sh

# 3. Deploy NATS
kubectl apply -f k8s/nats/

# 4. Deploy RBAC and Discovery
kubectl apply -f k8s/rbac/
kubectl apply -f k8s/discovery.yaml

# 5. Deploy inference and egress nodes
kubectl apply -f k8s/infer-detect.yaml
kubectl apply -f k8s/infer-classify.yaml
kubectl apply -f k8s/egress.yaml
```

Ingestor pods are spawned automatically by the Discovery node when sensor arrays appear on the ROS2 network.

---

## Hardware

Designed for **Raspberry Shake** devices (RS1D and RS4D) with an external 16-bit USB/I2S microphone for acoustic data. Deployed on Ubuntu 24.04 / `x86_64`. Developed on macOS.
