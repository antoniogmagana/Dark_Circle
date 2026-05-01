# inference-engine Helm chart

Helm chart for deploying the Dark Circle vehicle detection + classification
inference pipeline to a Kubernetes cluster. Subscribes to ROS2 sensor
topics, runs inference, publishes results back to a ROS2 output topic.

---

## Prerequisites

- Kubernetes 1.27+ (tested on kind v0.24)
- Helm 3.12+
- A container registry the cluster can pull from
- Optional: KEDA installed cluster-wide if you intend to enable
  autoscaling (`keda.enabled: true`). Install via
  `helm install keda kedacore/keda -n keda --create-namespace`.

The chart does NOT install KEDA, NATS clients, or ROS2 — only the
inference pipeline workloads themselves.

---

## Quick start

### 1. Build and publish the container images

The chart references six images by registry+tag. Build them from the
source tree, tag for your registry, and push:

```bash
cd inference-engine

# Build all six images locally (output: inference-engine/<name>:<tag>)
scripts/build_containers.sh

# Tag and push to your registry
for img in discovery ingestor infer-detect infer-classify egress fake-publisher; do
  docker tag inference-engine/$img:dev your-registry.example.com/$img:v0.1.0
  docker push your-registry.example.com/$img:v0.1.0
done
```

If your registry is private, create a pull secret in the target namespace:

```bash
kubectl create secret docker-registry registry-credentials \
  --docker-server=your-registry.example.com \
  --docker-username=<user> \
  --docker-password=<password>
```

### 2. Configure your install

Copy `values.yaml` to a site-specific file and edit:

```bash
cp chart/values.yaml my-site-values.yaml
$EDITOR my-site-values.yaml
```

Required edits at minimum:

- `images.registry` — set to your registry hostname
- `images.tag` — set to the tag you pushed
- `expectedSensors` — replace the example `shake-001` block with your
  actual sensor inventory and ROS2 topic names

If your registry is private:

- `images.pullSecrets: [{name: registry-credentials}]`

### 3. Install

```bash
helm install dark-circle ./chart -f my-site-values.yaml
```

Watch the rollout:

```bash
kubectl get pods -w
```

When all pods are `Running`, verify ROS2 input → inference → ROS2 output:

```bash
# Confirm Discovery sees your sensor topics
kubectl logs -l app=discovery --tail=20

# Confirm Ingestor pods spawned per sensor array
kubectl get pods -l app=ingestor

# Confirm inference results flow
kubectl exec deploy/egress -- /bin/bash -c \
  'source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash \
   && timeout 10 ros2 topic echo /inference_result' | head -40
```

---

## Configuration reference

All values are configurable via `values.yaml`. Defaults are tuned for the
Dark Circle smoke test environment and should be reviewed before a
production install.

### Container images

| Key | Default | Description |
|-----|---------|-------------|
| `images.registry` | `REPLACE_ME` | Registry prefix for all six images. **Required.** |
| `images.tag` | `latest` | Image tag. Use a versioned tag in production. |
| `images.pullPolicy` | `IfNotPresent` | Set to `Always` if you push under the same tag during dev. |
| `images.pullSecrets` | `[]` | List of `{name: ...}` pull secrets for private registries. |

### ROS2 / DDS

| Key | Default | Description |
|-----|---------|-------------|
| `ros2.domainId` | `0` | `ROS_DOMAIN_ID`. Pick unique per site to avoid crosstalk. |
| `ros2.rmwImplementation` | `rmw_fastrtps_cpp` | DDS middleware. `rmw_cyclonedds_cpp` also supported. |
| `ros2.outputTopic` | `/inference_result` | ROS2 topic for `InferenceResult` messages from egress. |
| `ros2.fastddsBuiltinTransports` | `UDPv4` | Forces FastDDS off shared-memory transport. Required for single-node clusters; safe everywhere. Empty = FastDDS default. |
| `ros2.fastddsProfile` | `""` | Optional FastDDS XML for unicast peers when multicast is blocked. Empty = built-in DDS defaults. |
| `ros2.hostNetwork` | `true` | Run ROS2 pods on host network. Required for kind / Calico-VXLAN; set `false` on multicast-capable CNIs (Cilium, Calico-IPIP). |

### Sensor whitelist

| Key | Default | Description |
|-----|---------|-------------|
| `expectedSensors` | one example `shake-001` | Map of `{array_id: {audio, seismic, accel?}}`. Array IDs become Deployment names (must be RFC 1123 subdomain). |

Each entry's `accel` block is optional but if present must define `x`,
`y`, AND `z`.

### Sensor signal parameters

| Key | Default | Description |
|-----|---------|-------------|
| `sensorConfig.audioBitDepth` | `16` | Acoustic ADC bit depth (signed). |
| `sensorConfig.seismicBitDepth` | `24` | Seismic ADC bit depth. |
| `sensorConfig.accelBitDepth` | `24` | Accelerometer ADC bit depth. |
| `sensorConfig.audioSampleRate` | `16000` | Acoustic samples/sec. |
| `sensorConfig.seismicSampleRate` | `100` | Seismic samples/sec. |
| `sensorConfig.accelSampleRate` | `100` | Accelerometer samples/sec/axis. |
| `sensorConfig.targetRate` | `16000` | Target rate for legacy upsampled mode. Ignored when `nativeRates: true`. |
| `sensorConfig.nativeRates` | `true` | Ship each channel at its native rate (CRL pipeline). Set `false` only for legacy CNN. |

### Inference workloads

Each of `inference.inferDetect`, `inference.inferClassify`,
`inference.egress` accepts:

| Key | Default | Description |
|-----|---------|-------------|
| `replicas` | `1` | Replica count when KEDA is disabled. |
| `minReplicas` | `1` | Lower bound when KEDA is enabled. |
| `maxReplicas` | `3` | Upper bound when KEDA is enabled. |
| `resources` | see values.yaml | Per-replica `requests` / `limits`. |

### Discovery / Ingestor

| Key | Default | Description |
|-----|---------|-------------|
| `discovery.resources` | small | Per-pod resources for the singleton Discovery controller. |
| `ingestor.resources` | medium | Per-pod resources for each Discovery-spawned Ingestor. |

### NATS / JetStream

| Key | Default | Description |
|-----|---------|-------------|
| `nats.enabled` | `true` | Deploy bundled NATS. Set `false` to use an external NATS cluster. |
| `nats.url` | `nats://nats-service:4222` | Override only when `enabled: false`. |
| `nats.image` | `nats:2.10` | Pinned for JetStream consumer-creation compatibility. |
| `nats.storage.type` | `emptyDir` | `emptyDir` (smoke test) or `pvc` (production). |
| `nats.storage.pvcSize` | `1Gi` | Used only when `type: pvc`. |
| `nats.storage.pvcStorageClass` | `""` | Empty = cluster default StorageClass. |

### KEDA autoscaling (optional, OFF by default)

| Key | Default | Description |
|-----|---------|-------------|
| `keda.enabled` | `false` | Emit ScaledObjects to scale infer-detect / infer-classify / egress on JetStream backlog. |
| `keda.lagThreshold` | `50` | Per-replica pending-message threshold. |
| `keda.pollingInterval` | `10` | Seconds between KEDA polls. |
| `keda.cooldownPeriod` | `60` | Seconds to wait before scaling down. |

### Fake publisher (smoke test only)

| Key | Default | Description |
|-----|---------|-------------|
| `fakePublisher.enabled` | `false` | Synthesize ROS2 sensor messages. **Disable in production.** |
| `fakePublisher.topics` | shake_001 fixtures | Comma-separated topic list. |
| `fakePublisher.rateHz` | `10` | Tick rate. |

---

## Architecture

```
ROS2 Network (Raspberry Shake sensors)
        │
        ▼
  Discovery Node ──── k8s API ────► Ingestor Pod (per sensor array)
                                            │ JetStream: sensor.data
                                            ▼
                                     Infer Detect Node
                                      │            │
                              detected=false    detected=true
                                      │            │ JetStream: detection.result
                                      │            ▼
                                      │      Infer Classify Node
                                      │            │ JetStream: classification.result
                                      ▼            ▼
                                        Egress Node
                                  (negatives via detection.result;
                                   positives via classification.result;
                                   one ROS2 publish per window)
                                             │ ROS2: <ros2.outputTopic>
                                             ▼
                                     InferenceResult msg
```

Discovery watches the ROS2 graph and spawns one Ingestor Deployment per
configured sensor array as soon as the array's required topics appear.
Ingestors normalize ADC samples, package 1-second windows, and publish
to NATS JetStream. The two inference nodes read from JetStream, run
TorchScript models, and publish results downstream. Egress bridges the
final result back to ROS2 on `<ros2.outputTopic>`.

---

## Upgrading

```bash
helm upgrade dark-circle ./chart -f my-site-values.yaml
```

Pods will roll automatically as their template hashes change. Note that
the JetStream init Job is a one-shot — re-running an upgrade does NOT
re-run it; if you change `expectedSensors` topology in a way that
requires new streams or consumers, delete and reinstall:

```bash
helm uninstall dark-circle
kubectl delete pvc -l app=nats   # only if nats.storage.type: pvc
helm install dark-circle ./chart -f my-site-values.yaml
```

---

## Uninstalling

```bash
helm uninstall dark-circle
```

The bundled NATS PVC (if `nats.storage.type: pvc`) and any sensor-array
Ingestor Deployments spawned by Discovery are deleted along with the
release. Manual cleanup is rarely needed.

---

## Known limitations

1. **Single-namespace.** Discovery's RBAC scopes it to its own namespace.
   Multi-namespace deployments need additional Role + RoleBinding pairs.

2. **No multi-cluster federation.** Each cluster runs an independent
   pipeline.

3. **JetStream emptyDir.** Default storage is in-memory; a NATS pod
   restart drops in-flight detection results. Switch to `nats.storage.type: pvc`
   for production.

4. **`FASTDDS_BUILTIN_TRANSPORTS=UDPv4` is set globally.** Necessary for
   single-node test clusters where ROS2 pods share `/dev/shm` mount
   namespaces. Real multi-host deployments don't need it but it's safe
   to leave enabled.
