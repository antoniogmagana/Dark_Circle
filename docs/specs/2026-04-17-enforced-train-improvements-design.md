# Enforced-Train Improvement Design
**Date:** 2026-04-17
**Mode:** Late-fusion multi-sensor vehicle detection/classification

---

## 1. Goal

Improve overall model performance across all three task modes (detection, category, instance) before selecting the best 1-audio + 1-seismic model per mode for the production late-fusion pipeline. The fusion layer (`evaluate_best_ensemble` in `eval.py`) is untouched — all improvements target model inputs and training procedure.

---

## 2. Scope

- **Sensors:** audio (16kHz) + seismic (200Hz)
- **Architectures:** all 6 (DetectionCNN, ClassificationCNN, WaveformClassificationCNN, ClassificationLSTM, InceptionTime, BiGRU) for both sensors — collapsed seismic 1D models retained for comparison
- **Modes:** detection (binary), category (4-class), instance (14-class)
- **Total runs:** 36 (6 architectures × 2 sensors × 3 modes)
- **Fusion layer:** unchanged — `evaluate_best_ensemble()` selects best 1 per sensor per mode by `val_f1`, combines with F1-weighted softmax

---

## 3. Root Causes Being Addressed

### 3a. Seismic mel spectrogram bottleneck
With `SAMPLE_SECONDS=1`, `N_FFT=128`, `HOP_LENGTH=32` at 200Hz:
- Window = 200 samples → ~7 mel frames → [B, 1, 64, 7] input
- Severely limits temporal modeling for 2D CNN models

### 3b. SNR augmentation mismatch
- `AUGMENT_SNR_RANGE=[10,30]`dB was designed for audio (16kHz, high-energy)
- Applying same range to seismic (200Hz, inherently low SNR) over-corrupts the signal

### 3c. Class imbalance in category and instance modes
- Inverse-frequency class weights capped at 10.0 — insufficient for rare instance classes
- No sampling-level guarantee that rare classes appear each epoch
- Early stopping at patience=8 cuts training before rare-class learning stabilizes

### 3d. Suboptimal throughput for available hardware
- `BATCH_SIZE=128`, `NUM_WORKERS` default, `TRAIN_STEPS_PER_EPOCH=50` leave H100 underutilized

---

## 4. Configuration Changes (`config.py`)

All new parameters are named constants with explicit defaults. No hardcoded values in component modules.

### 4a. Window and mel spectrogram
```python
SAMPLE_SECONDS = 2              # was 1 — doubles window for both sensors

# Global mel parameters — applied to both sensors
N_FFT = 64                      # was 128
HOP_LENGTH = 16                 # was 32
MEL_BINS = 64                   # unchanged
```

These are global values applied to both sensors. The primary motivation is seismic:
- Seismic: 400 samples, N_FFT=64, HOP_LENGTH=16 → ~25 mel frames → [B, 1, 64, 25] (was ~7 frames)
- Audio: 32000 samples, N_FFT=64, HOP_LENGTH=16 → ~2000 mel frames — no ceiling concern, no degradation expected

The MD5 cache key in `dataset.py` already includes `SAMPLE_SECONDS`, `N_FFT`, and `HOP_LENGTH`, so the preload cache invalidates automatically on first run.

### 4b. Sensor-specific SNR augmentation
```python
AUGMENT_SNR_RANGE_AUDIO    = [10, 30]   # unchanged — appropriate for 16kHz
AUGMENT_SNR_RANGE_SEISMIC  = [20, 40]   # less aggressive — preserves low-SNR seismic signal
```

`data_generator.py` `augment_batch()` receives the sensor-appropriate range from the training loop. The sensor is known at config construction time, so this is a config-level branch, not a runtime branch.

### 4c. Class balance
```python
CLASS_WEIGHT_CAP          = 10.0        # was hardcoded — now tunable, default unchanged
USE_WEIGHTED_SAMPLER      = True        # new — enables WeightedRandomSampler
WEIGHTED_SAMPLER_STRENGTH = 1.0         # 1.0 = fully balanced, 0.0 = uniform
```

`WeightedRandomSampler` is constructed in `train.py` from per-class sample counts. Sampler length = `len(dataset)` (not `len(dataset) // BATCH_SIZE`) to avoid epoch truncation. Compatible with `NUM_WORKERS` via standard PyTorch DataLoader.

### 4d. Training duration
```python
MAX_EPOCHS            = 100     # was 50
EARLY_STOP_PATIENCE   = 15      # was 8
CHECKPOINT_METRIC     = "mcc"   # was "val_f1" — MCC doesn't mask per-class collapse
```

Val steps use MCC computed over the full validation set at epoch end (not per-batch average). The existing `val_f1` metric continues to be logged for `evaluate_best_ensemble()` model selection — only the early-stopping/checkpointing signal changes.

### 4e. Hardware optimization (H100, 120 logical cores, 1TB VRAM)
```python
BATCH_SIZE              = 1024
NUM_WORKERS             = 96            # saturates 120 logical cores, leaves 24 for OS/GPU comms
PREFETCH_FACTOR         = 16
PIN_MEMORY              = True
PERSISTENT_WORKERS      = True          # avoids worker respawn overhead between epochs
TRAIN_STEPS_PER_EPOCH   = 400           # 400 × 1024 = 409,600 samples/epoch
VAL_STEPS_PER_EPOCH     = 128
TORCH_COMPILE           = True          # torch.compile() with H100 backend
AMP_ENABLED             = True          # bfloat16 — H100 native, no loss scaling needed
AMP_DTYPE               = "bfloat16"
```

**AMP note:** Prior training used float16 which caused instability. bfloat16 has the same dynamic range as float32 and requires no loss scaling — behavior is different on H100. This is validated behavior for this hardware class.

---

## 5. Code Changes by File

### `config.py`
- Add all new constants listed in Section 4
- Replace hardcoded `snr_range` references with `AUGMENT_SNR_RANGE_AUDIO` / `AUGMENT_SNR_RANGE_SEISMIC`
- Replace hardcoded `weight_cap=10.0` with `CLASS_WEIGHT_CAP`

### `train.py`
- Thread `AUGMENT_SNR_RANGE_AUDIO` or `AUGMENT_SNR_RANGE_SEISMIC` into `augment_batch()` based on `config.SENSOR`
- Add `WeightedRandomSampler` construction after dataset load when `USE_WEIGHTED_SAMPLER=True`
- Pass sampler to `DataLoader` (mutually exclusive with `shuffle=True`)
- Add `PERSISTENT_WORKERS`, `PREFETCH_FACTOR`, `PIN_MEMORY` to DataLoader kwargs
- Wrap model in `torch.compile()` when `TORCH_COMPILE=True`
- Wrap forward pass in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` when `AMP_ENABLED=True`
- Change early stopping and checkpoint logic to use MCC (`CHECKPOINT_METRIC="mcc"`)
- Update `MAX_EPOCHS` and `EARLY_STOP_PATIENCE` references
- Continue logging `val_f1` for `evaluate_best_ensemble()` compatibility

### `data_generator.py`
- `augment_batch(x, snr_range)` — signature already takes `snr_range`; confirm it is passed from train.py, not defaulted internally

### `preprocess.py`
- `MelSpectrogram` is constructed with `N_FFT`, `HOP_LENGTH`, `MEL_BINS` from config — confirm these are not hardcoded in the transform constructor call

### `dataset.py`
- Confirm MD5 cache key includes `SAMPLE_SECONDS`, `N_FFT`, `HOP_LENGTH` (expected: yes, based on prior analysis)
- No structural changes needed

### `run_pipeline.sh`
- No changes — script already runs all model/sensor/mode combinations in parallel

---

## 6. What Is Not Changing

- Model architectures (`models.py`) — no changes
- Fusion layer (`eval.py` `evaluate_best_ensemble()`) — no changes
- Dataset loading / parquet parsing (`dataset.py`) — no structural changes
- Directory structure and artifact naming conventions
- `val_f1` logging (used by `evaluate_best_ensemble()` for model selection)

---

## 7. Post-Training Selection Criteria

After retraining, model selection for the production pipeline uses the existing `evaluate_best_ensemble()` logic:
- Picks best 1 model per sensor per mode by `val_f1` from saved `meta.pt` files
- Combines seismic + audio probabilities with F1-weighted softmax

Manual review: any model with MCC < 0.3 on the test set warrants investigation before including in the production ensemble, even if it ranks first for its sensor/mode.

---

## 8. Success Criteria

- Seismic InceptionTime and DetectionCNN: MCC improvement over current baselines (detection: 0.60, category: 0.35–0.45, instance: 0.19–0.39)
- Audio category and instance: reduced per-class accuracy spread (currently warthog ~1.5% vs motorcycle ~62.6% in instance mode)
- No seismic 1D model (BiGRU, ClassificationLSTM, WaveformClassificationCNN) regression below current collapsed baseline — if one surprises us positively, it becomes a candidate
- Ensemble MCC exceeds prior baseline: category 0.6152, instance 0.6483 (ensemble-train March 2026 reference)
