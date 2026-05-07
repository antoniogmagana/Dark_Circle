# Maintenance and Monitoring Plan
## Dark Circle Inference Engine

**Document Version:** 1.0  
**Last Updated:** April 27, 2026  
**Authors:** AI Technician Capstone Group 5

---

## 1. Introduction

### 1.1 Project Overview

The Dark Circle inference engine is a containerized, real-time vehicle detection and classification system developed for the Army Research Laboratory's Live, Virtual, and Constructive (LVC) Toolkit. The system processes multi-modal sensor data (acoustic and seismic) from Raspberry Shake devices to:

1. **Detect** the presence of vehicles vs. background noise (binary classification)
2. **Classify** detected vehicles into specific types (multi-class classification)

The inference pipeline consists of five microservices deployed on Kubernetes:
- **Discovery Node**: Dynamically spawns/tears down Ingestor pods based on active sensor arrays
- **Ingestor Nodes**: Buffer and normalize raw sensor data, publish to NATS
- **Infer Detect Node**: Runs vehicle detection model (binary classification)
- **Infer Classify Node**: Runs vehicle classification model (multi-class classification)
- **Egress Node**: Publishes inference results back to ROS2 network

### 1.2 Purpose and Scope

This Maintenance and Monitoring Plan provides operational guidelines for:
- Monitoring data quality and model performance in production
- Managing model updates and version control
- Ensuring system reliability and security
- Responding to incidents and degraded performance
- Maintaining deployment infrastructure and resources

This plan applies to the inference engine once deployed in operational or test environments where continuous monitoring is required.

---

## 2. Model Monitoring

### 2.1 Data Monitoring

#### 2.1.1 Input Data Quality Checks

The Ingestor nodes perform real-time validation on incoming ROS2
`std_msgs/String` messages carrying a bundled-channel JSON payload (one
message per array per timestep, all channels in a single document).

**Required Fields Validation (per JSON message):**
- `timestamp_unix`: Numeric, present (drop message if missing or non-numeric)
- `channels`: List, present (drop message if missing or wrong type)
- Each `channels[i].channel`: Must be a known tag in the cluster's
  `channels.yaml` ConfigMap (skip the entry, keep processing the rest)
- Each `channels[i].readings`: List, present (skip the entry on miss)
- `state`: Optional, log-only (counted as background / trigger / unknown)

**Per-message rate validation (soft):**
- If `channels[i].sampling_rate` disagrees with the configured
  `expected_rate` for that channel's role, log a warning and continue.
  This is non-fatal because partial / variable-cadence chunks are
  expected during normal operation. Watch for sustained warnings —
  they indicate firmware drift that the window-close guard may catch.

**Window-close rate validation (hard):**
- At the end of each 1-second window, the buffer compares realized
  samples vs. `window * expected_rate` for every active channel.
  Drift > ±1% drops the window with a clear log line:
  `[buffer:<role>] window dropped: rate mismatch (got N, expected M)`.
  Grep for this pattern as the primary signal that a sensor's effective
  rate has degraded out of tolerance.

**Buffer State Monitoring:**
- Each Ingestor maintains a 1-second window buffer
- Monitor buffer fill rate per channel (acoustic, seismic, accel_x/y/z)
- Alert on dropped-window log lines (rate mismatch above)
- Alert on timestamp synchronization failures across channels

**Key Metrics:**
| Metric | Threshold | Action |
|--------|-----------|--------|
| Messages with missing fields | > 0.1% | Log warning, discard message |
| Sampling rate deviation | > ±5% from expected | Alert operators, potential sensor malfunction |
| Buffer underruns | > 1 per minute | Investigate network latency or sensor dropout |
| Timestamp sync failures | > 0.5% | Check sensor array clock synchronization |

#### 2.1.2 Data Distribution Monitoring

**Statistical Drift Detection:**

Monitor normalized sensor data statistics (computed on 1-second windows):

```python
# Per channel statistics to track
- mean (expected ~0 after normalization)
- std (expected ~0.1-0.3 for typical vehicle signals)
- min/max (should remain in [-1, 1] after ADC normalization)
- signal energy (Σ x²)
```

**Drift Detection Strategy:**

1. **Baseline Statistics:** Establish baseline distributions during initial deployment using representative operational data
2. **Rolling Window Comparison:** Compare current 1-hour statistics against baseline using:
   - Kolmogorov-Smirnov test (p < 0.05 indicates distribution shift)
   - Energy ratio (current/baseline): flag if < 0.3 or > 3.0
3. **Alerting:**
   - **Minor drift**: Log when statistical tests indicate shift but system performance remains within tolerances
   - **Major drift**: Alert operators if detection accuracy drops below 80% (5% below target) over 100-window average

**Environmental Factors:**

Track known confounders that affect sensor data:
- Time of day (traffic patterns)
- Weather conditions (wind noise in acoustic channel)
- Seasonal changes (ground moisture affects seismic propagation)

Log these factors with predictions for future correlation analysis.

#### 2.1.3 Data Preprocessing Validation

**ADC Normalization Verification:**

Ingestor nodes normalize raw ADC counts to [-1, 1]:
```python
# Acoustic (16-bit): scale = 2^15 = 32768
# Seismic/Accel (24-bit): scale = 2^23 = 8388608
normalized = raw_data / scale
```

**Validation:**
- Spot-check: Random sample 0.1% of windows and verify normalization bounds
- Alert if any normalized values exceed [-1.1, 1.1] (accounting for float precision)

**Mel Spectrogram Validation (Classification Only):**

Classification model expects Mel spectrograms. Monitor:
- Frequency bin distribution (should show characteristic vehicle acoustic signatures)
- Spectrogram energy concentration (vehicle signals typically show low-freq dominance)
- NaN/Inf check: Alert immediately if spectrograms contain non-finite values

---

### 2.2 Model Performance Monitoring

#### 2.2.1 Key Performance Indicators (KPIs)

**Detection Model (Binary Classification):**

| Metric | Target | Alert Threshold | Critical Threshold |
|--------|--------|----------------|-------------------|
| Accuracy | ≥ 85% | < 82% (1-hour avg) | < 80% (sustained) |
| Recall (Detection Rate) | ≥ 85% | < 82% | < 80% |
| False Alarm Rate | ≤ 15% | > 18% | > 20% |
| Inference Latency | < 100 ms | > 120 ms | > 150 ms |

**Classification Model (Multi-class):**

| Metric | Target | Alert Threshold | Critical Threshold |
|--------|--------|----------------|-------------------|
| Top-1 Accuracy | ≥ 65% | < 62% (1-hour avg) | < 60% (sustained) |
| Top-3 Accuracy | ≥ 85% | < 80% | < 75% |
| Recall (per-class) | ≥ 65% | < 60% | < 55% |
| False Alarm Rate | ≤ 25% | > 28% | > 30% |
| Inference Latency | < 100 ms | > 120 ms | > 150 ms |

**Confidence Score Distribution:**

Monitor prediction confidence scores:
- Detection model: Track distribution of confidence for positive predictions
- Classification model: Track top-class confidence and confidence gap (top1 - top2)
- Alert if mean confidence drops below 0.70 for correct predictions (indicates model uncertainty)

#### 2.2.2 Performance Tracking Over Time

**Logging Strategy:**

Each inference node publishes structured logs with:

```python
{
    "timestamp": "2026-04-27T10:15:32.456Z",
    "sensor_array": "shake_001",
    "window_id": "uuid-string",
    "node": "infer-detect | infer-classify",
    "model_version": "v1.2.0",
    "prediction": "vehicle | forester | ...",
    "confidence": 0.87,
    "inference_time_ms": 45.3,
    "ground_truth": null  # Populated if available
}
```

**Aggregation and Dashboards:**

1. **Real-time Metrics (1-minute aggregates):**
   - Inference rate (predictions/sec)
   - Mean confidence score
   - P50, P95, P99 latency
   - Error rate (NATS publish failures, model exceptions)

2. **Hourly Performance Reports:**
   - Aggregate accuracy (if ground truth available from field reports)
   - Per-class detection counts (histogram of vehicle types)
   - Alert summary

3. **Daily/Weekly Trends:**
   - Model performance degradation curves
   - Sensor array availability/uptime
   - Resource utilization trends

**Ground Truth Collection:**

**Challenge:** Real-time ground truth is unavailable in production.

**Strategies:**
1. **Operator Feedback Loop:** Implement simple UI for field operators to flag incorrect predictions
   - "Correct" / "Incorrect" buttons in LVC visualization
   - Flagged examples stored for later analysis
2. **Periodic Field Tests:** Monthly controlled tests with known vehicle types to measure true accuracy
3. **Cross-validation with Other Sensors:** Correlate with visual/radar systems when available

#### 2.2.3 Performance Degradation Triggers

**Automated Retraining Triggers:**

| Condition | Verification Period | Action |
|-----------|-------------------|--------|
| Accuracy < 80% (detection) | Sustained for 24 hours | Trigger retraining investigation |
| Accuracy < 60% (classification) | Sustained for 24 hours | Trigger retraining investigation |
| False alarm rate > 20% | Sustained for 12 hours | Review data drift, consider threshold tuning |
| Operator feedback: >10% negative flags | Per 1000 predictions | Collect flagged samples for retraining |

**Manual Review Triggers:**

- New vehicle type encountered (not in training set): Operators report unknown vehicle class
- Seasonal transition: Review performance during spring thaw (affects seismic data)
- Sensor array redeployment: New geographic locations may have different noise profiles

---

### 2.3 Monitoring Infrastructure

#### 2.3.1 Logging and Metrics Stack

**Container Logging:**
- All pods write JSON-formatted logs to stdout/stderr
- Kubernetes captures logs via `kubectl logs`
- **Future:** Aggregate logs to centralized system (e.g., ELK stack: Elasticsearch, Logstash, Kibana)

**Metrics Collection:**
- **Prometheus:** Scrape custom metrics from inference nodes
  - Each node exposes `/metrics` endpoint (to be implemented)
  - Metrics: inference_time_histogram, prediction_confidence, nats_publish_success_total
- **Kubernetes Metrics Server:** Track pod-level CPU, memory, network usage

**Visualization:**
- **Grafana:** Dashboards for real-time monitoring
  - Inference pipeline health dashboard
  - Model performance dashboard (accuracy, latency, throughput)
  - Resource utilization dashboard (per-pod CPU/memory)

#### 2.3.2 Integration with Existing Infrastructure

**ROS2 Network:**
- Inference results published to `/inference_result` topic (ROS2)
- Other LVC components subscribe to this topic for downstream processing
- Monitor ROS2 topic health using `ros2 topic hz` and `ros2 topic echo`

**NATS Message Broker:**
- NATS deployed as Kubernetes service (`nats-service:4222`)
- Monitor NATS metrics: message rate, pending messages, connection count
- Use `nats` CLI for diagnostics: `nats server info`, `nats stream report`

**Kubernetes Cluster:**
- Deployed on ARL AI2C on-premise infrastructure
- kubectl access for deployment management
- **RBAC:** Discovery node uses ServiceAccount with permissions to create/delete Deployments

#### 2.3.3 Monitoring Frequency and Reporting

**Real-time Monitoring (Continuous):**
- Grafana dashboards for 24/7 operator visibility
- Automated alerts via Prometheus Alertmanager (to be configured)
  - Slack/email notifications for critical thresholds
  - PagerDuty integration for after-hours incidents

**Scheduled Reports:**
- **Daily:** Automated email summary of:
  - System uptime and error counts
  - Total predictions and vehicle detection rate
  - Any triggered alerts or anomalies
- **Weekly:** Performance review meeting with stakeholders
  - Model accuracy trends
  - Infrastructure capacity planning
  - Incident post-mortems

**Alerting Hierarchy:**

| Severity | Response Time | Escalation |
|----------|--------------|------------|
| **Info** | Next business day | Log only |
| **Warning** | Within 4 hours | Email to dev team |
| **Critical** | Within 1 hour | Page on-call engineer + email |
| **Emergency** | Immediate | Page multiple engineers + notify stakeholders |

---

## 3. Model Maintenance

### 3.1 Retraining and Updates

#### 3.1.1 Retraining Triggers

**Scheduled Retraining:**
- **Quarterly:** Retrain models on accumulated production data + operator feedback
- Purpose: Capture seasonal variations and gradual domain shifts

**Event-driven Retraining:**

1. **Performance Degradation:**
   - Detection accuracy < 80% sustained for 24 hours
   - Classification accuracy < 60% sustained for 24 hours

2. **New Vehicle Class:**
   - Operators report new vehicle type not in CLASS_MAP
   - Collect ≥ 100 labeled examples, retrain classification model

3. **Sensor Array Expansion:**
   - New geographic deployment location
   - Environmental acoustics may differ (urban vs. rural, different terrain)
   - Retrain on site-specific data

4. **Data Distribution Shift:**
   - KS-test indicates significant drift (p < 0.01) sustained for 7 days
   - Review data quality, retrain if legitimate environmental change

#### 3.1.2 Retraining Procedure

**Pipeline Location:** `/project-files/Dark_Circle/model-train/`

**Steps:**

1. **Data Collection:**
   ```bash
   # Collect production data from LVC database (PostgreSQL)
   # Query time range: last 3 months or since last retraining
   python scripts/export_production_data.py --start-date 2026-01-01 --end-date 2026-04-01
   ```

2. **Data Labeling:**
   - Incorporate operator feedback (flagged correct/incorrect predictions)
   - Manual labeling of new vehicle classes (if applicable)
   - Validate label quality: minimum 2 independent annotators for new data

3. **Training Configuration:**
   ```python
   # config.py
   MODEL_NAME = "DetectionCNN"  # or "ClassificationCNN"
   TRAINING_MODE = "detection"  # or "instance"
   EPOCHS = 50
   BATCH_SIZE = 32
   LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
   # Enable class weights for imbalanced datasets
   CLASS_WEIGHTS = compute_class_weights(train_dataset)
   ```

4. **Model Training:**
   ```bash
   cd /project-files/Dark_Circle/model-train/
   poetry run python train.py
   # Outputs to: saved_models/[TRAINING_MODE]/[MODEL_NAME]/[RUN_ID]/
   ```

5. **Validation on Hold-out Set:**
   ```bash
   poetry run python eval.py --model-path saved_models/.../best_model.pth
   # Verify: Detection accuracy ≥ 85%, Classification accuracy ≥ 65%
   ```

6. **A/B Testing (if applicable):**
   - Deploy new model to 10% of sensor arrays
   - Monitor for 7 days, compare performance to baseline
   - If performance ≥ baseline, proceed to full rollout

#### 3.1.3 Testing Before Deployment

**Automated Test Suite:**
```bash
cd /project-files/Dark_Circle/inference-engine/
pytest tests/ -v
# Must pass all 97 tests (excluding test_buffer.py if no GPU)
```

**Integration Tests with New Model:**
1. Copy new model weights to temporary MODEL_DIR
2. Run inference tests:
   ```bash
   MODEL_DIR=/tmp/new_model pytest tests/test_infer_detect.py -v
   MODEL_DIR=/tmp/new_model pytest tests/test_infer_classify.py -v
   ```
3. Verify:
   - Model loads successfully on target device (CUDA/MPS/CPU)
   - Inference time < 100 ms per window
   - Output format matches protobuf schema

**Offline Performance Validation:**
- Evaluate on representative test set (≥ 1000 samples)
- Compare metrics to previous model version
- Document results in model card (see Section 3.2.3)

**Shadow Mode Deployment (Optional):**
- Deploy new model as shadow instance (publishes to separate NATS subject)
- Compare predictions to production model for 48 hours
- Analyze discrepancies: if agreement > 95%, proceed to production

---

### 3.2 Version Control and Rollback

#### 3.2.1 Version Control System

**Model Artifacts:**

All trained models stored in structured directory:
```
/project-files/Dark_Circle/model-train/saved_models/
├── detection/
│   └── DetectionCNN/
│       └── run_2026-04-15_1345/
│           ├── best_model.pth      # PyTorch weights
│           ├── config.json         # Training hyperparameters
│           ├── metrics.csv         # Epoch-by-epoch performance
│           ├── confusion_matrix.png
│           └── class_weights.json  # For imbalanced datasets
└── instance/
    └── ClassificationCNN/
        └── run_2026-04-20_0830/
            └── ...
```

**Git Version Control:**
- All code (training pipeline, inference engine) tracked in Git
- Model weights (.pth files) NOT committed to Git (too large)
- **Git tags** for production model versions:
  ```bash
  git tag -a v1.2.0 -m "Detection model retrained with Q1 2026 data"
  ```

**Model Registry (Future Enhancement):**
- Use MLflow or similar for centralized model tracking
- Track: model version, training date, dataset version, performance metrics, deployment status

#### 3.2.2 Deployment Versioning

**Kubernetes Deployment Strategy:**

Each inference node Deployment specifies model version via environment variable:

```yaml
# k8s/infer-detect.yaml
spec:
  containers:
  - name: infer-detect
    image: registry.ai2c.local/dark-circle/infer-detect:latest
    env:
    - name: MODEL_VERSION
      value: "v1.2.0"
    - name: MODEL_DIR
      value: "/models/detection/v1.2.0"
    volumeMounts:
    - name: model-storage
      mountPath: /models
      readOnly: true
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-pvc
```

**Model Storage:**
- Models stored in Kubernetes PersistentVolume (shared across pods)
- Path structure: `/models/[detection|classification]/[version]/best_model.pth`
- New versions uploaded to PV, Deployments updated to reference new path

#### 3.2.3 Rollback Procedures

**Trigger Conditions:**

Rollback to previous model version if:
1. New model accuracy < previous model accuracy - 5% (sustained for 6 hours)
2. Inference latency > 150 ms sustained
3. Critical bug discovered (e.g., model crashes on specific input patterns)

**Rollback Process:**

1. **Identify Previous Stable Version:**
   ```bash
   # Check Git tags for last production deployment
   git tag -l
   # Example: v1.1.0 was last stable
   ```

2. **Update Kubernetes Deployment:**
   ```bash
   # Edit MODEL_VERSION environment variable
   kubectl set env deployment/infer-detect MODEL_VERSION=v1.1.0
   kubectl set env deployment/infer-classify MODEL_VERSION=v1.1.0
   
   # Or apply updated YAML
   kubectl apply -f k8s/infer-detect.yaml
   kubectl apply -f k8s/infer-classify.yaml
   ```

3. **Verify Rollback:**
   ```bash
   # Check pod logs for model loading confirmation
   kubectl logs deployment/infer-detect -f
   # Expected: "Loaded model from /models/detection/v1.1.0/best_model.pth"
   
   # Monitor metrics for 1 hour to confirm performance restored
   ```

4. **Post-rollback Analysis:**
   - Document reason for rollback
   - Investigate root cause of new model failure
   - Update testing procedures to catch similar issues in future

**Rollback Time Target:** < 15 minutes from decision to full deployment

#### 3.2.4 Documentation of Model Updates

**Model Card Template:**

For each production model version, maintain a model card:

```markdown
# Model Card: DetectionCNN v1.2.0

## Model Details
- **Architecture:** DetectionCNN (3-layer CNN + 2 FC layers)
- **Training Date:** April 15, 2026
- **Trained By:** Brandon Taylor, Larry Parrotte
- **Git Commit:** abc123def
- **Dataset:** Q1 2026 production data + M3NVC + MOD_vehicle

## Intended Use
- **Primary Use:** Binary vehicle detection (vehicle vs. background noise)
- **Primary Users:** ARL LVC Toolkit operators
- **Out-of-scope Uses:** Not validated for indoor environments or non-ground vehicles (aircraft)

## Performance
| Metric | Value | Test Set |
|--------|-------|----------|
| Accuracy | 87.3% | 2500 samples, 50/50 balanced |
| Recall | 86.8% | |
| False Alarm Rate | 12.7% | |

## Training Data
- **Sources:** M3NVC dataset (5 sensor arrays × 24 hours), MOD_vehicle, production feedback
- **Size:** 125,000 1-second windows
- **Class Distribution:** 50% vehicle, 50% background

## Limitations
- Performance degrades in high-wind conditions (wind noise in acoustic channel)
- Lower accuracy for bicycles (softer acoustic signature)

## Ethical Considerations
- Model developed for military simulation and training purposes
- Potential dual-use concerns if deployed for autonomous weapons systems

## Changelog from v1.1.0
- Added 15,000 production samples with operator feedback
- Increased model capacity: 32 → 64 filters in first conv layer
- Improved false alarm rate from 15.2% to 12.7%
```

**Change Log Communication:**

- Post model updates to team Slack channel
- Email summary to stakeholders (mentors, ARL points of contact)
- Update inference engine README with current production versions

---

## 4. Operational Considerations

### 4.1 Resource Management

#### 4.1.1 Infrastructure Requirements

**Per-node Resource Allocation:**

| Node | CPU (cores) | Memory (GB) | GPU | Instances |
|------|-------------|-------------|-----|-----------|
| Discovery | 0.5 | 1 | No | 1 (singleton) |
| Ingestor | 1.0 | 2 | No | Dynamic (1 per sensor array) |
| Infer Detect | 2.0 | 4 | Yes (optional) | 1-3 (autoscale) |
| Infer Classify | 2.0 | 4 | Yes (optional) | 1-3 (autoscale) |
| Egress | 0.5 | 1 | No | 1 |
| NATS | 1.0 | 2 | No | 1 (or 3 for HA) |

**GPU Utilization:**
- Models support CUDA, MPS (Apple Silicon), and CPU fallback
- GPU recommended for production: 4-10× speedup over CPU
- Single NVIDIA GPU (e.g., RTX 3070) sufficient for up to 10 sensor arrays
- GPU shared across inference pods using NVIDIA device plugin for Kubernetes

**Storage Requirements:**
- **Model Storage:** ~500 MB per model version (store 3-5 versions)
- **Persistent Volume:** 5 GB for model artifacts
- **Logs:** ~1 GB/day (compressed), retain 30 days
- **Database:** Dependent on LVC PostgreSQL instance (not managed by inference engine)

#### 4.1.2 Scaling Strategy

**Horizontal Pod Autoscaling (HPA):**

Infer nodes scale based on NATS queue depth:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: infer-detect-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: infer-detect
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: External
    external:
      metric:
        name: nats_pending_messages
        selector:
          matchLabels:
            subject: sensor.data
      target:
        type: AverageValue
        averageValue: "100"  # Scale up if > 100 messages pending
```

**Ingestor Dynamic Scaling:**

Discovery node automatically spawns/tears down Ingestor pods:
- 1 Ingestor pod per active sensor array
- Discovery polls ROS2 network every 30 seconds
- Ingestor torn down if no messages received for 5 minutes (sensor offline)

**Resource Limit Enforcement:**

```yaml
resources:
  requests:  # Guaranteed allocation
    cpu: 1000m
    memory: 2Gi
  limits:    # Maximum allowed
    cpu: 2000m
    memory: 4Gi
```

Prevents resource contention and ensures fair sharing in multi-tenant K8s cluster.

#### 4.1.3 Cost Optimization

**On-premise Deployment:** No cloud costs; primary cost is hardware capital expense.

**Optimization Strategies:**

1. **Model Compression:**
   - Quantization: Convert FP32 models to INT8 (PyTorch quantization)
   - Target: 2-4× speedup, 2-3× memory reduction, < 1% accuracy loss
   - **Not yet implemented** – future enhancement

2. **Batch Inference (if latency allows):**
   - Process multiple 1-second windows in parallel
   - Requires modifying inference nodes to queue windows
   - Trade-off: 20-30% throughput gain vs. 2-3× increased latency

3. **CPU-only Deployment for Low-traffic Scenarios:**
   - If < 3 sensor arrays active, disable GPU allocation
   - Reduces power consumption and GPU wear

4. **Idle Resource Reclamation:**
   - Ingestors automatically terminate when sensors offline
   - Manual scale-down of inference nodes during maintenance windows

---

### 4.2 Security and Compliance

#### 4.2.1 Data Privacy and Access Control

**Sensor Data Sensitivity:**
- Acoustic and seismic data may indirectly reveal information about personnel movements
- Data classified as **Controlled Unclassified Information (CUI)** under DoD guidelines

**Access Control:**

1. **Kubernetes RBAC:**
   - Inference engine pods run under dedicated ServiceAccounts with minimal permissions
   - Discovery node: `create`, `get`, `list`, `delete` on Deployments only
   - Other nodes: No K8s API access
   - Developers: `kubectl` access restricted to `dark-circle` namespace

2. **NATS Security:**
   - Currently: No authentication (trusted internal network)
   - **Recommended:** Enable NATS authentication with client certificates
   - Restrict publish/subscribe permissions per node:
     - Ingestor: Publish to `sensor.data` only
     - Infer Detect: Subscribe to `sensor.data`, publish to `detection.result`
     - Etc.

3. **Model Artifact Security:**
   - PersistentVolume mounted read-only in pods (prevents tampering)
   - Upload access restricted to CI/CD pipeline or authorized personnel

**Data Retention:**
- Sensor data not persisted by inference engine (streaming only)
- Logs retained for 30 days, then archived or deleted
- Models retained indefinitely (but old versions archived after 1 year)

#### 4.2.2 Regulatory Compliance

**Applicable Regulations:**

1. **DoD Cybersecurity Requirements:**
   - System deployed on Army network must comply with DISA STIGs
   - Regular vulnerability scanning (Nessus, OpenSCAS)
   - Patch management: OS and container base images updated quarterly

2. **ITAR (International Traffic in Arms Regulations):**
   - Vehicle classification capability may be ITAR-controlled
   - No foreign national access without export license
   - Code repository hosted on .mil or approved .gov infrastructure

3. **ATO (Authority to Operate):**
   - Required before deployment on operational Army networks
   - Security assessment by ARL Chief Information Security Officer (CISO)
   - Documentation: System Security Plan (SSP), Risk Assessment, Contingency Plan

**Compliance Verification:**

- Annual security audit by ARL CISO
- Continuous monitoring via ACAS (Assured Compliance Assessment Solution)
- Incident reporting to Army Cyber Command within 1 hour

#### 4.2.3 Secure Development Practices

**Container Image Security:**

1. **Base Image Selection:**
   - Use official ROS2 images: `ros:jazzy` (Ubuntu 24.04)
   - Scan images with Trivy: `trivy image ros:jazzy`
   - No critical vulnerabilities in base image

2. **Minimal Image Size:**
   - Multi-stage Dockerfile: Build dependencies discarded in final image
   - Only runtime dependencies included (PyTorch, protobuf, NATS client)

3. **Image Signing:**
   - **Recommended:** Sign images with Docker Content Trust (DCT)
   - Ensures integrity: Only authorized builds deployed

**Dependency Management:**

- Poetry for Python dependencies (lockfile ensures reproducibility)
- Regular dependency updates: `poetry update` monthly
- Vulnerability scanning: `poetry audit` (future: integrate into CI/CD)

**Secrets Management:**

- Currently: No secrets (NATS unauthenticated, no external APIs)
- **Future:** Use Kubernetes Secrets for NATS credentials
  - Mounted as environment variables, NOT baked into images

---

### 4.3 Incident Management

#### 4.3.1 Incident Classification

| Severity | Definition | Examples |
|----------|------------|----------|
| **P1 - Critical** | Complete system failure, no predictions | NATS broker down, all infer pods crashing |
| **P2 - High** | Degraded performance, critical alerts | Accuracy < 75%, latency > 200 ms sustained |
| **P3 - Medium** | Non-critical component failure | Single Ingestor pod crash, Discovery node restart loop |
| **P4 - Low** | Minor issues, no user impact | Warning logs, transient NATS connection failures |

#### 4.3.2 Incident Response Procedures

**Detection:**
1. **Automated Alerts:** Prometheus Alertmanager → Slack/PagerDuty
2. **User Reports:** LVC operators report anomalies via email/Slack
3. **Monitoring Dashboard:** On-call engineer reviews Grafana dashboards every 4 hours

**Initial Response (within SLA):**

| Severity | Response Time | Initial Actions |
|----------|--------------|-----------------|
| P1 | 15 minutes | On-call engineer paged, stakeholders notified |
| P2 | 1 hour | On-call engineer investigates, updates status channel |
| P3 | 4 hours | Assigned to on-call engineer, triaged during business hours |
| P4 | Next business day | Added to backlog, monitored for escalation |

**Diagnostic Checklist:**

```bash
# 1. Check pod health
kubectl get pods -n dark-circle
kubectl describe pod <pod-name>

# 2. Review recent logs
kubectl logs <pod-name> --tail=100

# 3. Check NATS broker
kubectl exec -it nats-0 -- nats server info
kubectl exec -it nats-0 -- nats stream report

# 4. Verify ROS2 network connectivity
kubectl exec -it <ingestor-pod> -- ros2 topic list
kubectl exec -it <ingestor-pod> -- ros2 topic hz /raw_sensor_reading

# 5. Check resource utilization
kubectl top pods -n dark-circle
kubectl top nodes

# 6. Review recent configuration changes
git log --since="24 hours ago" --oneline
kubectl rollout history deployment/infer-detect
```

#### 4.3.3 Escalation Procedures

**Escalation Path:**

1. **On-call Engineer** (L1):
   - Initial triage and diagnostics
   - Attempt standard remediation (restart pods, scale resources)
   - Escalate if issue not resolved in 2 hours (P1) or 8 hours (P2)

2. **Senior Engineer / Tech Lead** (L2):
   - Deep debugging (code-level investigation)
   - Emergency patches or hotfixes
   - Escalate to model developers if issue related to model behavior

3. **Model Developers** (L3):
   - Model-specific issues (accuracy degradation, unexpected predictions)
   - Emergency model rollback or retraining

4. **Project Mentors / Stakeholders** (Executive):
   - Major incidents affecting mission-critical operations
   - Decisions on system downtime or architectural changes

**Communication Channels:**

- **Slack:** `#dark-circle-incidents` (real-time updates)
- **Email:** Weekly incident summary to stakeholders
- **Status Page:** (Future) Public status page for LVC Toolkit components

#### 4.3.4 Mitigation and Resolution

**Common Incidents and Remediation:**

| Incident | Root Cause | Immediate Fix | Long-term Prevention |
|----------|------------|---------------|----------------------|
| All Ingestor pods down | Discovery node RBAC permissions expired | Reapply RBAC manifests | Implement RBAC certificate rotation monitoring |
| NATS broker unresponsive | Message queue overflow (slow consumers) | Restart NATS, increase max messages | Tune NATS stream limits, add HPA for infer nodes |
| Inference pod OOM crash | Large batch size or memory leak | Increase memory limit, restart pod | Profile memory usage, fix leak or reduce batch size |
| Latency > 200 ms | CPU throttling (resource contention) | Scale up CPU requests | Move inference pods to dedicated GPU nodes |
| Accuracy drops to 70% | New vehicle type not in training set | Document new class, plan retraining | Implement "unknown class" detection and alerting |

**Fallback Mechanisms:**

1. **Graceful Degradation:**
   - If Classify node fails, Egress still publishes Detection-only results
   - LVC Toolkit receives partial intelligence (vehicle detected, type unknown)

2. **Manual Override:**
   - Operators can manually annotate detections in LVC UI if model predictions clearly wrong
   - Manual annotations logged for later retraining

3. **Circuit Breaker Pattern (Future Enhancement):**
   - If inference node consistently fails (e.g., model crashes on certain inputs), skip inference and pass through raw data
   - Prevents cascade failures

#### 4.3.5 Post-incident Review

**Within 3 business days of P1/P2 incident resolution:**

1. **Incident Report Template:**
   ```markdown
   ## Incident Summary
   - **Date/Time:** 2026-04-20 14:35 UTC
   - **Duration:** 2 hours 15 minutes
   - **Severity:** P2
   - **Affected Systems:** Infer Classify node, 3 sensor arrays
   
   ## Timeline
   - 14:35: Alert triggered - Latency > 150 ms
   - 14:40: On-call engineer begins investigation
   - 15:10: Root cause identified - GPU memory leak
   - 15:45: Hotfix deployed (restart pods with reduced batch size)
   - 16:50: Performance confirmed restored
   
   ## Root Cause
   - Memory leak in Mel spectrogram computation (librosa issue)
   - GPU memory gradually filled, causing OOM and restarts
   
   ## Resolution
   - Temporary: Reduced batch size from 8 to 4
   - Permanent: Upgrade librosa to v0.10.2 (leak fixed)
   
   ## Lessons Learned
   - Need GPU memory monitoring alerts
   - Test GPU memory usage in long-running scenarios
   
   ## Action Items
   - [Owner: John] Add GPU memory metric to Grafana (Due: 2026-04-25)
   - [Owner: Antonio] Update CI/CD to include 1-hour soak test (Due: 2026-05-01)
   ```

2. **Knowledge Base Update:**
   - Add incident to troubleshooting runbook
   - Update incident response checklist with new diagnostic steps

3. **Review Meeting:**
   - 30-minute blameless post-mortem with team
   - Focus on systemic improvements, not individual fault

---

## 5. Documentation and Knowledge Transfer

### 5.1 Documentation Standards

**Existing Documentation:**

1. **Project README** (`/project-files/Dark_Circle/README.md`):
   - High-level project overview, architecture, objectives

2. **Inference Engine README** (`inference-engine/README.md`):
   - Node descriptions, message flow, build/deploy instructions
   - Kept up-to-date with architecture changes

3. **Test Documentation** (`inference-engine/tests/README.md`):
   - Comprehensive test suite documentation (97 tests)
   - Test categories, coverage, run instructions

4. **Model Training README** (`model-train/README.md`):
   - Training pipeline configuration guide
   - Model registry, hyperparameters, dataset preparation

5. **This Maintenance Plan** (`inference-engine/MAINTENANCE_MONITORING_PLAN.md`)

**Required Updates:**

- **Daily:** Incident log updates (if applicable)
- **Weekly:** Performance report summaries
- **After each deployment:** Update README with new model versions, configuration changes
- **After each retraining:** Update model card, commit to Git

### 5.2 Knowledge Transfer Strategy

**Target Audiences:**

1. **ARL Operational Team:** LVC Toolkit operators who monitor system in production
2. **ARL Maintenance Team:** Engineers responsible for long-term maintenance post-handover
3. **Future Capstone Teams:** Students who may extend or enhance the system

**Training Plan:**

#### 5.2.1 For ARL Operational Team (2-hour workshop)

**Topics:**
- LVC Toolkit integration: How inference results appear in simulation
- Monitoring dashboards: Key metrics and how to interpret them
- Basic troubleshooting: When to restart pods, when to escalate
- Operator feedback loop: How to flag incorrect predictions

**Hands-on Activities:**
- Walkthrough of Grafana dashboards
- Simulated incident response (guided pod restart)

**Deliverables:**
- Quick-reference laminated card: "Inference Engine Health Checklist"
- 15-minute recorded video walkthrough

#### 5.2.2 For ARL Maintenance Team (8-hour training)

**Day 1 (4 hours): System Architecture and Deployment**
- Deep dive into microservices architecture (Discovery, Ingestor, Infer, Egress)
- ROS2, NATS, Kubernetes fundamentals
- Live deployment walkthrough: `kubectl apply -f k8s/`
- Code tour: Key files in `src/` directory

**Day 2 (4 hours): Monitoring, Retraining, and Incident Response**
- Setting up Prometheus + Grafana
- Model retraining walkthrough: `train.py` to production deployment
- Incident response role-playing: Simulated NATS failure
- Q&A and documentation review

**Deliverables:**
- Access credentials for K8s cluster and Git repository
- "Runbook" document: Step-by-step procedures for common tasks
- Contact information for Capstone team (6-month support period)

#### 5.2.3 Code Documentation

**Inline Comments:**
- Docstrings for all public functions (Google-style)
- Inline comments for complex logic (e.g., buffer synchronization in Ingestor)

**Example:**
```python
def normalize_sensor_data(raw_data: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Normalize ADC counts to [-1, 1] range.
    
    Args:
        raw_data: Raw ADC counts from sensor
        bit_depth: ADC resolution (16 for acoustic, 24 for seismic)
    
    Returns:
        Normalized data in [-1, 1] range
    
    Note:
        Scale factor is 2^(bit_depth - 1) to handle signed integers.
        For 16-bit: scale = 32768, for 24-bit: scale = 8388608.
    """
    scale = 2 ** (bit_depth - 1)
    return raw_data / scale
```

**README Clarity:**
- Each node directory (`src/{node}/`) has a README explaining:
  - Purpose and responsibilities
  - Configuration environment variables
  - Dependencies and how to test locally

### 5.3 Transition and Handover Plan

**Timeline:**

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1: Documentation** | 2 weeks | Finalize all documentation, record training videos |
| **Phase 2: Training** | 1 week | Conduct training workshops for ARL teams |
| **Phase 3: Shadow Support** | 4 weeks | Capstone team available for questions, ARL team takes primary responsibility |
| **Phase 4: Full Handover** | Ongoing | ARL team fully autonomous, Capstone team on-call for emergencies only (6 months) |

**Handover Checklist:**

- [ ] All documentation reviewed and approved by ARL
- [ ] Training sessions completed, attendance confirmed
- [ ] kubectl access and credentials transferred to ARL maintenance team
- [ ] Git repository access granted to ARL team
- [ ] Monitoring dashboards configured and accessible to ARL team
- [ ] Contact information exchanged (Capstone team emergency contact)
- [ ] Security assessment completed, ATO granted (if applicable)
- [ ] System deployed in production environment and stable for 30 days

**Support Agreement:**

- **Months 1-3 post-handover:** Capstone team responds to questions within 24 hours
- **Months 4-6 post-handover:** Best-effort support, 72-hour response time
- **After 6 months:** No guaranteed support, but team may assist on volunteer basis

---

## 6. Conclusion

### 6.1 Summary

This Maintenance and Monitoring Plan provides a comprehensive framework for ensuring the long-term reliability, performance, and security of the Dark Circle inference engine. The plan addresses:

- **Model Monitoring:** Real-time tracking of data quality and model performance against target KPIs (85% detection accuracy, 65% classification accuracy)
- **Model Maintenance:** Structured retraining procedures triggered by performance degradation or new requirements
- **Operational Reliability:** Resource management, scaling strategies, and security best practices
- **Incident Management:** Clear escalation procedures and response protocols for system failures
- **Knowledge Transfer:** Comprehensive documentation and training for ARL teams

### 6.2 Key Success Factors

1. **Proactive Monitoring:** Automated alerts catch issues before they impact mission-critical operations
2. **Continuous Improvement:** Quarterly retraining incorporates production feedback and evolving operational environments
3. **Robust Testing:** 97-test suite ensures high confidence in deployments and updates
4. **Clear Ownership:** Well-defined roles and escalation paths for incident response
5. **Comprehensive Documentation:** Enables ARL teams to operate independently post-handover

### 6.3 Future Enhancements

**Near-term (3-6 months):**
1. **Centralized Logging:** Deploy ELK stack for log aggregation and analysis
2. **Prometheus Metrics:** Instrument all nodes with custom metrics exporters
3. **Grafana Dashboards:** Create operational dashboards for real-time monitoring
4. **NATS Authentication:** Enable TLS and client certificates for message broker security
5. **Model Registry:** Deploy MLflow for centralized model versioning and tracking

**Medium-term (6-12 months):**
1. **A/B Testing Framework:** Automated canary deployments for model updates
2. **Model Compression:** Quantization to reduce latency and resource usage
3. **Anomaly Detection:** ML-based detection of unusual sensor data patterns (early warning for data drift)
4. **Operator Feedback UI:** Web interface for flagging incorrect predictions (currently manual process)

**Long-term (1-2 years):**
1. **Federated Learning:** Multi-site training without centralizing sensitive sensor data
2. **Continuous Learning:** Online learning from production data (with human-in-the-loop validation)
3. **Multi-modal Fusion:** Incorporate additional sensor types (radar, infrared) if available
4. **Explainable AI:** Visualizations showing which frequencies/signals drove each prediction (enhance operator trust)

### 6.4 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model performance degrades over time | Medium | High | Quarterly retraining, continuous monitoring |
| New vehicle types not in training data | Medium | Medium | Operator feedback loop, rapid retraining capability |
| Infrastructure failure (K8s cluster down) | Low | High | Deploy NATS in HA mode, maintain spare hardware |
| Data distribution shift (new deployment location) | Medium | Medium | Site-specific retraining, transfer learning |
| Security breach (unauthorized model access) | Low | High | RBAC enforcement, regular security audits |
| Loss of institutional knowledge (team turnover) | High | Medium | Comprehensive documentation, training, 6-month support period |

---

## 7. Appendices

### Appendix A: Configuration Reference

**Environment Variables (All Nodes):**

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `NATS_URL` | `nats://nats-service:4222` | NATS broker address | |
| `SENSOR_ARRAY` | Required | Unique sensor array ID | `shake_001` |
| `MODEL_DIR` | `/models` | Directory containing model weights | `/models/detection/v1.2.0` |
| `MODEL_VERSION` | Required | Model version identifier | `v1.2.0` |
| `LOG_LEVEL` | `INFO` | Logging verbosity | `DEBUG`, `INFO`, `WARNING` |
| `NAMESPACE` | `default` | Kubernetes namespace (Discovery only) | `dark-circle` |

### Appendix B: Monitoring Metrics Catalog

**Detection Model Metrics:**
- `detection_accuracy`: Rolling 1-hour accuracy (requires ground truth)
- `detection_recall`: True positive rate
- `detection_false_alarm_rate`: False positive rate
- `detection_confidence_mean`: Average confidence for positive predictions
- `detection_inference_time_ms`: P50, P95, P99 latency

**Classification Model Metrics:**
- `classification_accuracy_top1`: Top-1 accuracy
- `classification_accuracy_top3`: Top-3 accuracy
- `classification_recall_per_class`: Recall for each vehicle class
- `classification_confidence_mean`: Average top-class confidence
- `classification_inference_time_ms`: P50, P95, P99 latency

**System Health Metrics:**
- `ingestor_buffer_fill_rate`: % of time buffer has full 1-second window
- `nats_pending_messages`: Queue depth per subject
- `pod_cpu_usage`: CPU utilization per pod
- `pod_memory_usage`: Memory utilization per pod
- `pod_restarts_total`: Cumulative restart count (high value indicates instability)

### Appendix C: Contact Information

**Capstone Team (Emergency Support):**
- Brandon Taylor: btaylor@cmu.edu
- Larry Parrotte: lparrotte@cmu.edu
- John Tomaselli: jtomaselli@cmu.edu
- Antonio Magana: amagana@cmu.edu

**ARL Mentors / Stakeholders:**
- Dr. Kristin E. Schaefer-Lay: kristin.e.schaefer-lay.civ@army.mil
- Dr. Damon Conover: damon.conover.civ@army.mil
- Henry Reimert: henry.reimert.ctr@army.mil

**Escalation for Critical Incidents:**
- ARL AI2C Operations Center: [Contact TBD by ARL]

---

**Document Control:**
- **Version:** 1.0
- **Approved by:** [To be signed by Project Mentor]
- **Next Review Date:** July 27, 2026 (quarterly review)
- **Change Log:**
  - v1.0 (2026-04-27): Initial version

