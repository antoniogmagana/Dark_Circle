# Live, Virtual, and Constructive (LVC) Toolkit

**Project Codename:** Dark Circle
**Team:** AI Technician Capstone Group 5, Carnegie Mellon University, ARL
**Team Members:** Brandon Taylor, Larry Parrotte, John Tomaselli, Antonio Magana
**Mentors:** Dr. Kristin E. Schaefer-Lay, Dr. Damon Conover, Henry Reimert

---

## 1. Project Overview

This project is a Capstone effort for the Carnegie Mellon University AI Technician Program, developed in collaboration with the Army Research Laboratory (ARL). The goal is to extend ARL's Live, Virtual, and Constructive (LVC) Toolkit with AI/ML techniques that turn raw acoustic and seismic sensor streams into actionable, real-time vehicle intelligence.

The repository contains two cooperating components:

* **`crl-train/`** — a Causal Representation Learning (CRL) framework that learns a multi-modal latent representation from paired audio (16 kHz) and seismic (100 Hz) sensor windows, then trains downstream probe heads for vehicle **presence** (binary) and vehicle **type** (multi-class). Trained models are exported as TorchScript bundles.
* **`inference-engine/`** — a containerized ROS2 + NATS JetStream + Kubernetes pipeline that loads those bundles and produces near real-time classifications from live or replayed sensor feeds, publishing results back into the LVC environment.

## 2. Problem Statement

In modern military operations, the ability to detect and identify potential threats early and accurately is a decisive advantage. Visual identification is often hindered by terrain, foliage, and adversary camouflage, creating a critical need for non-visual detection methods.

ARL faces a capability gap in the automated interpretation of data from tactical edge sensors. While the LVC Toolkit can aggregate raw sensor data, it lacks the robust AI mechanisms to transform that data into actionable intelligence. Any technical solution must be:

* **Lightweight and Power-Efficient:** to run on soldier-carried or embedded devices.
* **Resilient:** to function in austere field conditions with limited connectivity.
* **Integratable:** to work with existing Army tactical systems.

## 3. Solution Architecture

The system is split into a **training side** and an **inference side**, connected by a TorchScript bundle artifact.

```text
            ┌────────────────────────────┐       export_for_inference.py       ┌────────────────────────────────┐
raw_data ──▶│  crl-train/                │ ──────────────────────────────────▶ │  inference-engine/             │
            │  (CRL + downstream probes) │   detect-bundles/, classify-bundles/│  (ROS2 + NATS + Kubernetes)    │
            └────────────────────────────┘                                     └────────────────────────────────┘
                                                                                          │
                                                                              live sensors / LVC autonomy stack
```

* **Training (`crl-train/`)** — produces `crl_best.pth` and downstream head checkpoints under `saved_crl/runs/<frontend>/<training_mode>/<run-id>/`. `export_for_inference.py` fuses an encoder + a head into a self-contained TorchScript bundle.
* **Bridge** — bundles drop into `inference-engine/detect-bundles/` (presence head) or `inference-engine/classify-bundles/` (type head). Each catalog tracks a `*-default` symlink that the pods load.
* **Inference (`inference-engine/`)** — the pods have **no Python dependency on `crl_vehicle`**; they only call `torch.jit.load()` on the bundle. This keeps the runtime image small and decouples deploys from training-code churn.

![Solution Concept Diagram](https://github.com/antoniogmagana/Dark_Circle/blob/main/images/solution-concept.png)

![Architecture Diagram](https://github.com/antoniogmagana/Dark_Circle/blob/main/images/architecture.png)

## 4. Scientific Hypothesis

The core of our research is based on the following empirical hypothesis:

> Using ARL-approved seismic and acoustic data, a **multi-modal classification model** will perform better than the most promising single-mode classification models for identifying selected vehicles from each other and from random background noise.

#### Target Performance Metrics

| Task | Metric | Target |
| :--- | :--- | :--- |
| **Vehicle vs. Background Noise** | Accuracy | ≥ 85% |
| | Recall (Pd) | ≥ 85% |
| | False Alarm Rate | ≤ 15% |
| **Vehicle vs. Other Vehicles** | Accuracy | ≥ 65% |
| | Recall (Pd) | ≥ 65% |
| | False Alarm Rate | ≤ 25% |

## 5. Training Framework (`crl-train/`)

A modular CRL framework with pluggable components, all wired through a single `CRLConfig` dataclass:

* **Frontends:** `multiscale` (fixed Conv1D, three kernel sizes, early fusion) plus four Morlet variants — `morlet_per_sensor`, `morlet_fused`, `morlet_learnable`, `morlet_learnable_fused` (fixed vs. learnable wavelet scales × per-sensor vs. early fusion).
* **Training modes:** `vae` (ELBO with adaptive β annealing), `disentangled` (signal/env latent split with invariance + cross-modal alignment losses), `contrastive` (NT-Xent over stratified positive/negative pairs).
* **Priors:** `standard` (N(0, I)) or `conditional` (iVAE-style label conditioning).
* **Downstream probes:** linear, MLP, or full-latent heads for presence and type, trained on a frozen representation.
* **Two-stage learnable Morlet:** loads a converged fixed-Morlet checkpoint and fine-tunes the learnable variant with reduced LR on filters.
* **Export:** `export_for_inference.py` converts a trained run into TorchScript detect or classify bundles consumed by the inference engine.

### Quick start

```bash
cd crl-train
poetry install
# Full pipeline: CRL + all downstream probe variants + cross-dataset eval.
poetry run python run_full_diagnostic.py --frontend multiscale --crl-epochs 100
```

For the full CLI (per-frontend overrides, sweep harness via `run_experiments.py`, post-hoc analysis scripts, two-stage Morlet, leaderboard generation), see [`crl-train/README.md`](crl-train/README.md). Math reference for ELBO / KL / NT-Xent: [`crl-train/docs/math_reference.md`](crl-train/docs/math_reference.md).

## 6. Inference Engine (`inference-engine/`)

A five-node Kubernetes pipeline that consumes TorchScript bundles and produces real-time detections + classifications:

* **Discovery** — polls the ROS2 graph and spawns/tears down per-sensor-array Ingestor deployments.
* **Ingestor** — subscribes to ROS2 sensor topics (JSON-string `std_msgs/String` payloads), maintains a 1-second window buffer, and publishes to NATS. ADC scaling is gated behind `ADC_SCALE_NORMALIZE` (default off — raw counts, matching the CRL training contract).
* **Infer Detect** — loads a detect bundle (encoder + presence head) and emits `DetectionResult` on NATS.
* **Infer Classify** — loads a classify bundle (encoder + type head), re-encodes from raw waveforms, and emits `EgressPayload` on NATS.
* **Egress** — re-publishes inference results onto a ROS2 topic for downstream consumers, with `publish_time` and `latency_seconds` instrumentation.

Transport is NATS JetStream with three durable streams (`SENSOR_DATA`, `DETECTION_RESULT`, `CLASSIFICATION_RESULT`). Container images are built from a `ros:jazzy` base via `scripts/build_containers.sh`; deployment ships as both raw k8s manifests and a Helm chart.

### Quick start

```bash
cd inference-engine
# On a kind cluster, build images first so pods don't ImagePullBackOff.
scripts/build_containers.sh
helm install dark-circle ./chart -f my-site-values.yaml
```

For the chart values reference, multi-site DDS configuration, smoke and replay tooling, and operational runbooks, see [`inference-engine/README.md`](inference-engine/README.md), [`inference-engine/chart/README.md`](inference-engine/chart/README.md), and [`inference-engine/MAINTENANCE_MONITORING_PLAN.md`](inference-engine/MAINTENANCE_MONITORING_PLAN.md).

## 7. From Trained Run to Deployed Bundle

`export_for_inference.py` is the only link between the two sides. From `crl-train/`:

```bash
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind {detect,classify} \
    --bundle-name <frontend>-<mode>-<run>[-<probe>]-v<N>
```

The output lands in `inference-engine/detect-bundles/` or `inference-engine/classify-bundles/`. Pods pick a bundle via the `DETECT_BUNDLE` / `CLASSIFY_BUNDLE` environment variables (or the `*-default` symlink). Bundle naming and selection rules: [`inference-engine/detect-bundles/README.md`](inference-engine/detect-bundles/README.md), [`inference-engine/classify-bundles/README.md`](inference-engine/classify-bundles/README.md).

## 8. Project Status & Roadmap

* [x] Requirements gathering and design
* [x] Data engineering and CRL modeling (`crl-train/`)
* [x] Inference engine architecture and implementation (`inference-engine/`)
* [x] Training → bundle export pipeline (`export_for_inference.py`)
* [ ] Deployment and integration at customer sites
* [ ] Evaluation and redeployment

Per-node implementation status, test counts, and verification matrices live in [`inference-engine/README.md`](inference-engine/README.md).

---

## 9. Data Management

The two pipelines consume different formats. Both store **raw ADC counts** (no normalization — see §6 on `ADC_SCALE_NORMALIZE`); they diverge on schema and layout.

### Training format (`crl-train/`)

The training pipeline reads pre-windowed **Parquet** or **CSV** files (whichever you have on disk) produced by an upstream parsing step like `server-load/sample_parse.py`, which converts raw captures and adds per-sample presence labels via a 3-gate detector. Parquet is strongly preferred — see "Parquet vs. CSV" below.

**Layout** — one file per (dataset × sensor × vehicle × sensor-array node):

```text
data_files/parsed/
├── train/
│   ├── focal_audio_walk_rs1.parquet      # or .csv
│   ├── focal_seismic_walk_rs1.parquet    # or .csv
│   ├── iobt_audio_polaris0150pm_rs1.parquet
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Filename pattern: `{dataset}_{modality}_{vehicle}_{rs_node}.{parquet,csv}`, where:

* `dataset` ∈ `{focal, iobt, m3nvc}` — source rates are hardcoded per-dataset in `crl_vehicle/data/dataset.py` (audio 16 kHz / seismic 100 Hz for `focal`/`iobt`; audio 1.6 kHz / seismic 200 Hz for `m3nvc`).
* `modality` ∈ `{audio, seismic}`.
* `vehicle` is the class label key (e.g. `walk`, `pickup`, `cx30`, `background`) — mapped to a target class via `DATASET_VEHICLE_MAP` in `crl_vehicle/config.py`.

**Required columns** (one row = one sample at source rate; identical schema for parquet and CSV):

| Column | Type | Required for |
| :--- | :--- | :--- |
| `amplitude` | `float32` (or int) | always — raw ADC value |
| `present` | `bool` | always — per-sample ground-truth presence flag |
| `scene_id` | `int64` | only when the dataset/vehicle uses the `split_runs` marker |
| `run_id` | `int64` | only when the dataset/vehicle uses the `split_runs` marker |

Windows and resampling to canonical target rates happen on first load and are cached as `.pt` tensors under `saved_crl/caches/`. The 1-second presence label is a per-window majority vote on the `present` column.

**Parquet vs. CSV.** Both formats produce bit-identical training inputs, but parquet is dramatically more efficient: O(1) row counts via footer metadata, column-scoped reads (`present` reads touch only the bool column on disk), and 5–15× smaller on-disk footprint. CSV reads scan the whole file, so the **first** epoch and **first** manifest build on a CSV-only dataset are noticeably slower; subsequent runs are absorbed by the `.pt` cache. **If both `foo.parquet` and `foo.csv` exist for the same stem, the loader uses parquet and warns** — keep the source of truth in one format to avoid drift.

### Replay format (`inference-engine/scripts/replay_publisher.py`)

The replay publisher emits the same JSON message format the live ingestor expects, ticked at 10 Hz over a 1-second window. It accepts three file formats; no fixed directory layout — paths are passed directly.

| Format | Required columns / properties | Sample rate source |
| :--- | :--- | :--- |
| **Parquet** | `amplitude` (float/int), `time_stamp` (float, seconds); `present` (bool, optional, enables per-window scoring) | inferred from median `time_stamp` delta, or override with `--audio-rate` / `--seismic-rate` |
| **CSV** | same as Parquet; header row required unless `--no-header` (positional: amplitude, time_stamp, present) | same as Parquet |
| **WAV** | mono PCM, 8/16/24/32-bit | read from WAV header; no `present`, so scoring is skipped |

Audio and seismic are supplied as separate files at matching start/end times:

```bash
poetry run python scripts/replay_publisher.py \
    --audio  /path/to/audio.parquet \
    --seismic /path/to/seismic.parquet
```

The published `std_msgs/String` JSON envelope carries `sensor_id`, `timestamp_unix`, and a `channels[]` array; each channel has a `channel` tag (e.g. `MIC`, `EHZ`, `ENN`), a `sampling_rate`, and a `readings[]` list. Channel tags are resolved to roles (`acoustic`, `seismic`, `accel_{x,y,z}`) via the deployed `channels.yaml` ConfigMap. Acoustic + seismic are required; the three accelerometer axes are optional but all-or-nothing.

### Where to get the data

* **Cloud Storage (recommended):** [shared link — TODO]
* **Git LFS:** `git lfs install && git lfs pull` (if your team has LFS configured).
* **Team Access:** contact the CMU Capstone Team Group 5 directly.

> [!NOTE]
> The legacy `raw_data/<experiment>/rs1/{aud16000.csv, aud16k.wav, ehz.csv, gps.csv}` layout is what `Data exploration.ipynb` consumes. It is **not** the training or replay format — those use the pre-windowed `{amplitude, present, ...}` schema documented above. (Note that the legacy "CSV" name overlap is incidental — the legacy files have completely different columns.)

## 10. Getting Started

Each sub-project has its own Poetry-based install. Pick the one that matches what you want to do:

* **Train models** — see [`crl-train/README.md`](crl-train/README.md). Install with `cd crl-train && poetry install`, then run `poetry run python run_full_diagnostic.py ...` or `poetry run python train.py ...`.
* **Run the inference engine** — see [`inference-engine/README.md`](inference-engine/README.md) for the node-level walkthrough and [`inference-engine/chart/README.md`](inference-engine/chart/README.md) for the Helm chart quick-start.
* **Explore the raw data interactively** — the legacy `Data exploration.ipynb` at the repo root still uses `pip install -r requirements.txt` for the notebook's own dependencies. It is not required for training or inference.
