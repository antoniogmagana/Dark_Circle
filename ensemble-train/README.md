# LVC Ensemble Training Pipeline

Sensor-fusion vehicle detection and classification using ground-truth vibration data from IOBT, FOCAL, and M3NVC datasets. Trains individual models per sensor type, then fuses predictions via a weighted ensemble.

## Architecture

The pipeline follows a two-stage ensemble approach:

**Stage 1 — Detection:** One model per sensor (audio, seismic, accel) votes on whether a vehicle is present. Softmax scores are fused via weighted average.

**Stage 2 — Classification:** When a vehicle is detected, a second set of per-sensor models votes on the vehicle type. Supports category-level classification (pedestrian, light, sport, utility) and instance-level identification (per-vehicle identity).

Each sensor trains independently because the signal characteristics are fundamentally different:

| Sensor   | Sample Rate | Samples/sec | Channels | Signal Character |
|----------|-------------|-------------|----------|------------------|
| Audio    | 16,000 Hz   | 16,000      | 1        | Rich frequency content up to 8kHz |
| Seismic  | 100-200 Hz  | 200         | 1        | Low-frequency ground vibration, 1-80Hz |
| Accel    | 100-200 Hz  | 200         | 3 (x,y,z)| Directional vibration patterns |

## Requirements

- Python 3.12+
- PostgreSQL with the `lvc_db` database populated
- CUDA-capable GPU (recommended)
- Poetry for dependency management

```bash
poetry install
```

Set the database password before running:

```bash
export DB_PASSWORD=your_password
```

## Quick Start

### Run the full pipeline interactively

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

The script prompts you to select sensors, training modes, and model architectures, then runs training → evaluation → ensemble build in sequence.

### Run a single training job

```bash
TRAIN_SENSOR=seismic TRAINING_MODE=detection MODEL_NAME=ResNet1D \
    poetry run python train.py
```

### Evaluate all trained models

```bash
poetry run python eval.py
poetry run python aggregate_results.py
```

### Build and evaluate the ensemble

```bash
poetry run python ensemble.py build
poetry run python ensemble.py eval
poetry run python ensemble.py show
```

## File Reference

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters, dataset constants, sensor profiles |
| `models.py` | Model architectures (6 models) |
| `train.py` | Training loop with early stopping, LR scheduling, gradient clipping |
| `eval.py` | Per-model evaluation, generates reports + confusion matrices |
| `ensemble.py` | Weighted late fusion across sensor models |
| `preprocess.py` | Per-window z-score normalization + mel spectrogram extraction |
| `dataset.py` | PostgreSQL-backed dataset with block-based deterministic splits |
| `data_generator.py` | Synthetic background generation + SNR augmentation |
| `db_utils.py` | PostgreSQL connection and query helpers |
| `aggregate_results.py` | Collects all evaluation reports into a leaderboard CSV |
| `run_pipeline.sh` | Interactive batch runner for the full pipeline |

## Available Models

### 2D Models (Mel Spectrogram Input)

**DetectionCNN** — Lightweight 2-layer CNN designed for binary detection. Small parameter count, fast inference. Best suited for audio where the spectrogram is information-rich.

**ClassificationCNN** — Deeper CNN (2-4 layers depending on sensor) for multi-class problems. Uses AdaptiveAvgPool2d so it works on any spectrogram size.

### 1D Models (Raw Waveform Input)

**WaveformClassificationCNN** — 3-layer 1D CNN operating directly on the time-domain signal. Kernel sizes and strides adapt to sensor sample rate.

**ClassificationLSTM** — CNN front-end for feature extraction followed by a multi-layer LSTM for temporal modelling. Good at capturing sequential patterns in vibration signatures.

**ResNet1D** — Residual 1D CNN with skip connections. Particularly effective for short signals (seismic/accel at 200 samples) where skip connections preserve gradient flow through limited temporal extent. Architecture: stem → residual stages with stride-2 downsampling → global average pool → head.

**IterativeMiniRocket** — Frozen random convolutional kernel features (from the tsai library) with a trainable linear head. Very fast to train since only the head is optimized.

## Model Tuning

This is the primary workflow for improving results. All hyperparameters live in `config.py` in the `_HYPERPARAMS` dictionary, keyed by `(MODEL_NAME, TRAIN_SENSOR)`.

### Where to change hyperparameters

Open `config.py` and find the `_HYPERPARAMS` dictionary (Section 10). Each entry looks like:

```python
("ResNet1D", "seismic"): {
    "LEARNING_RATE": 1e-3,
    "CHANNELS": [32, 64, 128],
    "STEM_KERNEL": 7,
    "STEM_STRIDE": 1,
    "BLOCKS_PER_STAGE": 2,
    "HIDDEN": 64,
    "DROPOUT": 0.3,
},
```

Every model+sensor combination has its own profile. Change one without affecting others.

### What each hyperparameter controls

**Shared across all models:**

- `LEARNING_RATE` — Adam optimizer learning rate. Start at 1e-3, reduce to 1e-4 if training is unstable.
- `HIDDEN` — Classifier hidden layer size. Larger = more capacity but more overfitting risk.
- `DROPOUT` — Dropout rate in the classifier. Increase (0.4-0.5) if overfitting, decrease (0.1-0.2) if underfitting.

**2D models (DetectionCNN, ClassificationCNN):**

- `CHANNELS` — Output channels per conv layer. The list length determines the network depth. Example: `[32, 64]` is 2 layers, `[32, 64, 128, 256]` is 4 layers.
- `KERNELS` — Kernel size per layer. Use 3 for most cases, 5 for the first layer if you want a wider receptive field.
- `STRIDES` — Convolution stride per layer. Usually 1 (let MaxPool handle downsampling).
- `PADS` — Padding per layer. Typically `kernel // 2` to preserve spatial dimensions.
- `POOL` — Whether to add MaxPool2d(2,2) after each conv block. Set `True` for audio (large spectrograms), `False` for seismic/accel (small spectrograms — AdaptiveAvgPool handles the final pooling).

**1D models (WaveformClassificationCNN):**

- `CHANNELS` — Same as 2D: list length = depth.
- `KERNELS` — Kernel sizes. For audio (16K samples): large kernels like `[64, 32, 16]`. For seismic (200 samples): small kernels like `[7, 5, 3]`.
- `STRIDES` — Convolution strides. Audio can afford aggressive strides `[8, 4, 2]`. Seismic needs conservative strides `[2, 2, 1]` — the signal is already short.

**ClassificationLSTM:**

- `CHANNELS`, `KERNELS`, `STRIDES` — CNN front-end (same logic as 1D models above).
- `POOLS` — MaxPool1d kernel per CNN layer.
- `LAYERS` — LSTM depth. 2-3 is typical. More layers = more capacity but slower.
- `DIM` — Intermediate classifier dimension between LSTM output and final prediction.

**ResNet1D:**

- `CHANNELS` — Channel progression: `[stem, stage1, stage2, ...]`. Each stage doubles channels and halves temporal resolution.
- `STEM_KERNEL` — Initial convolution kernel. Larger for audio (15), smaller for seismic (7).
- `STEM_STRIDE` — Initial stride. Audio uses 4 (quick downsampling from 16K), seismic uses 1 (preserve all 200 samples).
- `BLOCKS_PER_STAGE` — Residual blocks per stage. 2 is a good default. Increase for more depth without changing channel progression.

**IterativeMiniRocket:**

- `MINIROCKET_FEATURES` — Number of random convolutional kernels. Default 1000. Increasing to 5000-10000 may help but slows feature extraction.

### Training controls in config.py

These are in Section 7-8 and apply to all runs:

```python
EPOCHS = 30              # Max epochs (early stopping usually triggers before this)
EARLY_STOP_PATIENCE = 8  # Stop after 8 epochs without improvement
GRAD_CLIP = 1.0          # Gradient clipping max norm (stabilizes LSTM training)
BEST_MODEL_METRIC = "val_f1"  # Which metric to optimize: "val_f1", "val_acc", "val_loss"
BATCH_SIZE = 128         # Reduce to 64 if you run out of GPU memory
```

### Data augmentation controls

In Section 9 of `config.py`:

```python
SYNTHESIZE_BACKGROUND = True     # Generate synthetic background samples during training
SYNTHESIZE_PROBABILITY = 0.5     # Probability of replacing a background sample with synthetic
AUGMENT_SNR = True               # Add random noise to training samples
AUGMENT_SNR_RANGE = (10, 30)     # SNR range in dB (lower = more noise)
OVERSAMPLE_BACKGROUNDS = True    # Balance background vs vehicle samples in detection mode
```

### Class weights

In Section 5, these balance the loss function for imbalanced classes:

```python
# Detection mode
CLASS_WEIGHTS = [25.5, 1.0]  # [background, vehicle] — high weight on background
                              # because there are far more vehicle samples

# Category mode
CLASS_WEIGHTS = [28.0, 2.0, 13.0, 15.5]  # [pedestrian, light, sport, utility]
```

Adjust these based on your dataset's class distribution. Higher weight = model penalized more for misclassifying that class.

### Tuning strategy

**Start narrow, expand later.** Don't run all 54 combinations at once. Pick one sensor and one mode:

```bash
TRAIN_SENSOR=seismic TRAINING_MODE=detection MODEL_NAME=ResNet1D \
    poetry run python train.py
```

Watch the console output. Key signals:

- **Train acc high, val acc low** → Overfitting. Increase DROPOUT, reduce HIDDEN, reduce CHANNELS.
- **Both train and val acc low** → Underfitting. Increase CHANNELS (more capacity), add layers, increase HIDDEN.
- **Val loss jumps around** → Learning rate too high, or gradient instability. Reduce LEARNING_RATE to 1e-4, ensure GRAD_CLIP is 1.0.
- **LR reduced message appears early** → The scheduler is firing because val F1 plateaued. This is normal and usually helps.
- **Early stopping at epoch 5-6** → Model converged quickly (good for simple tasks) or is stuck (check if val acc is reasonable).

**Compare across architectures for the same sensor+mode:**

```bash
for model in ResNet1D WaveformClassificationCNN ClassificationLSTM; do
    TRAIN_SENSOR=seismic TRAINING_MODE=detection MODEL_NAME=$model \
        poetry run python train.py
    sleep 1
done
poetry run python eval.py
poetry run python aggregate_results.py
```

The leaderboard in `saved_models/master_evaluation_results.csv` will show you which model won.

## Directory Structure After Training

```
saved_models/
├── detection/
│   ├── audio/
│   │   ├── DetectionCNN/
│   │   │   └── 20260319_083432/
│   │   │       ├── best_model.pth
│   │   │       ├── hyperparameters.json
│   │   │       ├── meta.pt
│   │   │       ├── metrics.csv
│   │   │       ├── evaluation_report.txt
│   │   │       ├── predictions.npz
│   │   │       └── conf_matrix.png
│   │   ├── ResNet1D/
│   │   │   └── ...
│   │   └── ...
│   ├── seismic/
│   │   └── ...
│   └── accel/
│       └── ...
├── category/
│   └── (same structure)
├── instance/
│   └── (same structure)
├── master_evaluation_results.csv
└── ensemble_weights.json
```

## Ensemble Usage

### Building the ensemble

After training and evaluation, the ensemble discovers the best model per (mode, sensor) pair based on F1 score:

```bash
poetry run python ensemble.py build
```

This creates `saved_models/ensemble_weights.json` with the selected models and their fusion weights.

### Viewing the configuration

```bash
poetry run python ensemble.py show
```

Prints which model was selected for each sensor and mode, along with its F1 score and fusion weight.

### Evaluating

```bash
poetry run python ensemble.py eval
```

Runs weighted fusion on test predictions and compares ensemble accuracy against individual models.

### Programmatic inference

For integration into a live inference engine:

```python
from ensemble import SensorEnsemble

# Load pre-built ensemble
ens = SensorEnsemble.load()

# Two-stage prediction: detect → classify
result = ens.two_stage_predict({
    "seismic": seismic_tensor,  # [C, T] raw waveform
    "audio": audio_tensor,       # [C, T] raw waveform
})

# result = {
#     "detected": True,
#     "detection_confidence": 0.94,
#     "category_class_id": 2,
#     "category_confidence": 0.81,
#     "category_probs": array([...]),
# }
```

Missing sensors are handled automatically — if only seismic data is available, the ensemble renormalizes weights over the available sensors and still produces a prediction.

## Preprocessing

All data goes through the same preprocessing pipeline (`preprocess.py`) at both training and inference time:

1. **DC offset removal** — subtract the window mean (removes sensor drift)
2. **Per-window z-score** — divide by the window's standard deviation (normalizes to zero mean, unit variance)
3. **Spike clipping** — clamp to ±10 standard deviations (limits transient noise)
4. **Mel spectrogram** — (2D models only) convert to frequency domain

This pipeline is self-contained — no external calibration data is needed, which simplifies deployment.

## Database Schema

The pipeline expects a PostgreSQL database (`lvc_db`) with tables named:

```
{dataset}_{sensor}_{instance}_{node}
```

For example: `focal_seismic_mustang_n01`, `iobt_audio_warhog1135am_s01`

Each table has columns:
- `time_stamp` (float) — sample timestamp
- `amplitude` (float) — for audio and seismic
- `accel_x_ew`, `accel_y_ns`, `accel_z_ud` (float) — for accelerometer
- `run_id` (int, nullable) — used by M3NVC dataset for multi-run tables