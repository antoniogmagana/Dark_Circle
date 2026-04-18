# Enforced-Train Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve model convergence and per-class accuracy across all 36 training runs (6 architectures × 2 sensors × 3 modes) before selecting the best 1-audio + 1-seismic model per mode for the production late-fusion pipeline.

**Architecture:** All changes are configuration and training-loop only — model architectures and the fusion layer are untouched. The five improvements are: (1) 2s windows, (2) explicit mel params tuned for seismic, (3) sensor-specific SNR augmentation, (4) WeightedRandomSampler + raised weight cap, (5) MCC-based checkpointing with longer patience.

**Tech Stack:** Python, PyTorch, torchaudio, scikit-learn, enforced-train/{config.py, train.py, dataset.py}

---

## File Map

| File | Change |
|------|--------|
| `enforced-train/config.py` | Replace dynamic N_FFT/HOP_LENGTH with explicit constants; add AUGMENT_SNR_RANGE_AUDIO/SEISMIC, CLASS_WEIGHT_CAP, USE_WEIGHTED_SAMPLER, WEIGHTED_SAMPLER_STRENGTH, CHECKPOINT_METRIC, EARLY_STOP_PATIENCE, AMP_ENABLED, AMP_DTYPE, TORCH_COMPILE; update BATCH_SIZE, EPOCHS, NUM_WORKERS, PREFETCH_FACTOR, TRAIN_STEPS_PER_EPOCH, VAL_STEPS_PER_EPOCH |
| `enforced-train/train.py` | Wire AUGMENT_SNR_RANGE sensor branch; replace EpochShuffleSampler with WeightedRandomSampler when enabled; update DataLoader kwargs; add torch.compile; add AMP autocast; add MCC computation; switch checkpoint/early-stop to CHECKPOINT_METRIC; wire EARLY_STOP_PATIENCE from config; enforce TRAIN_STEPS_PER_EPOCH / VAL_STEPS_PER_EPOCH; pass CLASS_WEIGHT_CAP to _compute_class_weights |
| `enforced-train/dataset.py` | Add N_FFT and HOP_LENGTH to the MD5 cache key in `_get_cache_path()` |

---

## Task 1: Explicit mel parameters and window size in config.py

**Files:**
- Modify: `enforced-train/config.py:211-252`

Currently `N_FFT` and `HOP_LENGTH` are computed dynamically from `SEQ_LEN` (lines 249–252). This makes them impossible to set independently. Replace with explicit constants. Also update `SAMPLE_SECONDS`, `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, throughput settings, and add all new constants.

- [ ] **Step 1: Replace SAMPLE_SECONDS, EPOCHS, BATCH_SIZE, NUM_WORKERS, TRAIN_STEPS_PER_EPOCH, VAL_STEPS_PER_EPOCH**

In `enforced-train/config.py`, change lines 18, 211–213 from:
```python
BATCH_SIZE = 128
EPOCHS = 50
NUM_WORKERS = 32
...
TRAIN_STEPS_PER_EPOCH = 50
VAL_STEPS_PER_EPOCH = 16

# Time-scale knobs
SAMPLE_SECONDS = 1
```
to:
```python
BATCH_SIZE = 1024
EPOCHS = 100
NUM_WORKERS = 96
...
TRAIN_STEPS_PER_EPOCH = 400
VAL_STEPS_PER_EPOCH = 128

# Time-scale knobs
SAMPLE_SECONDS = 2
```

- [ ] **Step 2: Replace dynamic N_FFT/HOP_LENGTH with explicit constants**

In `enforced-train/config.py`, replace lines 247–252:
```python
# FFT window size — largest power of 2 ≤ SEQ_LEN, capped at 1024 for audio.
# Guarantees n_fft // 2 < SEQ_LEN (STFT padding constraint).
# Seismic (SEQ_LEN=200): N_FFT=128. Audio (SEQ_LEN=16000): N_FFT=1024.
N_FFT = min(1024, 1 << (SEQ_LEN.bit_length() - 1))

# Hop length between STFT frames (~25% of N_FFT, standard convention)
HOP_LENGTH = N_FFT // 4
```
with:
```python
# Explicit mel parameters — global for both sensors.
# Tuned for seismic (200Hz × 2s = 400 samples → ~25 mel frames).
# Audio (16kHz × 2s = 32000 samples) produces ~2000 frames — no ceiling concern.
N_FFT = 64
HOP_LENGTH = 16
```

- [ ] **Step 3: Add augmentation, class balance, training, and hardware constants**

After the `AUGMENT_SNR_RANGE` line (currently line 233), replace:
```python
AUGMENT_SNR_RANGE = (10, 30)
```
with:
```python
# Sensor-specific SNR ranges
AUGMENT_SNR_RANGE_AUDIO    = (10, 30)   # appropriate for 16kHz high-energy signal
AUGMENT_SNR_RANGE_SEISMIC  = (20, 40)   # less aggressive for low-SNR 200Hz signal
AUGMENT_SNR_RANGE = AUGMENT_SNR_RANGE_AUDIO if TRAIN_SENSOR == "audio" else AUGMENT_SNR_RANGE_SEISMIC
```

Then after the `AUGMENT_SNR_RANGE` block, add the following new constants block (insert before section 8 MODEL HYPERPARAMETERS):
```python
# =====================================================================
# 7b. CLASS BALANCE
# =====================================================================
CLASS_WEIGHT_CAP          = 10.0   # inverse-frequency weight ceiling (tunable)
USE_WEIGHTED_SAMPLER      = True   # WeightedRandomSampler for train DataLoader
WEIGHTED_SAMPLER_STRENGTH = 1.0    # 1.0=fully balanced, 0.0=uniform

# =====================================================================
# 7c. TRAINING PROCEDURE
# =====================================================================
EARLY_STOP_PATIENCE = 15           # was hardcoded 8 in train.py
CHECKPOINT_METRIC   = "mcc"        # "mcc" | "val_f1" | "val_loss" | "val_acc"

# =====================================================================
# 7d. HARDWARE & PRECISION
# =====================================================================
PREFETCH_FACTOR    = 16
PIN_MEMORY         = True
PERSISTENT_WORKERS = True
TORCH_COMPILE      = True
AMP_ENABLED        = True
AMP_DTYPE          = "bfloat16"    # H100 native — no loss scaling needed
```

- [ ] **Step 4: Verify SEQ_LEN still derives correctly**

`SEQ_LEN = int(REF_SAMPLE_RATE * SAMPLE_SECONDS)` at line 244 — with `SAMPLE_SECONDS=2` and seismic `REF_SAMPLE_RATE=200`, `SEQ_LEN=400`. With audio `REF_SAMPLE_RATE=16000`, `SEQ_LEN=32000`. Confirm no per-model kernel/stride computation breaks with the new SEQ_LEN values by reading lines 255–345 and verifying the `max()`/`//` guards still produce positive integers. No code change needed if guards hold.

- [ ] **Step 5: Verify INCEPTION_STEM_STRIDE is correct for new SAMPLE_SECONDS**

`INCEPTION_STEM_STRIDE = max(1, SEQ_LEN // 200)` (line 327):
- Seismic: `max(1, 400 // 200)` = 2. Previously 1. This now downsamples the seismic stem — acceptable since 400 samples still passes through.
- Audio: `max(1, 32000 // 200)` = 160. Previously 80. Confirm this is acceptable given the InceptionTime kernel sizes [9, 19, 39].

If the audio stem stride of 160 is too aggressive (post-stem length = 32000/160 = 200 samples, identical to before with 1s windows), this is fine — the post-stem length is preserved by design.

- [ ] **Step 6: Commit**

```bash
cd enforced-train
git add config.py
git commit -m "config: 2s windows, explicit mel params, sensor-specific SNR, class balance and training constants"
```

---

## Task 2: Update dataset.py cache key to include N_FFT and HOP_LENGTH

**Files:**
- Modify: `enforced-train/dataset.py:313-326`

The `_get_cache_path()` method builds an MD5 hash from dataset/sensor/sample_seconds/window_step/split/data_dir. It does not include `N_FFT` or `HOP_LENGTH`. Since changing these values affects the preloaded tensors' expected window sizes indirectly (via `SAMPLE_SECONDS`) but not the raw waveform cache itself — the cache stores raw waveform samples, not mel features. Raw waveform cache is keyed correctly by `sample_seconds` alone.

**However**, changing `SAMPLE_SECONDS` from 1→2 is already in the key, which will naturally bust the cache on first run. No key change needed for N_FFT/HOP_LENGTH since those only affect preprocessing (GPU-side, post-cache). Leave `_get_cache_path()` unchanged.

- [ ] **Step 1: Confirm cache key correctness**

Read `enforced-train/dataset.py` lines 313–326 and confirm `key_data` includes `sample_seconds`. Since mel transform runs in `preprocess_for_training()` on the GPU after the DataLoader yields raw waveforms, N_FFT/HOP_LENGTH do not affect what is stored in the cache. No change needed.

- [ ] **Step 2: Commit note**

No code change. Add a comment to `_get_cache_path()` to document this:
```python
def _get_cache_path(self):
    # Cache stores raw waveforms only. Mel params (N_FFT, HOP_LENGTH) are
    # applied GPU-side in preprocess_for_training() and do not affect this key.
    key_data = {
```

```bash
git add dataset.py
git commit -m "dataset: document cache key scope (raw waveforms only, mel params excluded)"
```

---

## Task 3: Wire sensor-specific SNR augmentation in train.py

**Files:**
- Modify: `enforced-train/train.py:54-56`

Currently `train_one_epoch` reads `config.AUGMENT_SNR_RANGE` directly. With Task 1 complete, `config.AUGMENT_SNR_RANGE` is already set to the sensor-appropriate value at config load time (the ternary in config.py). No change needed in train.py — `getattr(config, "AUGMENT_SNR_RANGE", (10, 30))` at line 55 picks up the correct value automatically.

- [ ] **Step 1: Confirm the existing wire-up is correct**

Read `enforced-train/train.py` lines 54–56:
```python
if getattr(config, "AUGMENT_SNR", False):
    current_snr_range = getattr(config, "AUGMENT_SNR_RANGE", (10, 30))
    x = augment_batch(x, snr_range=current_snr_range)
```
With Task 1 complete, `config.AUGMENT_SNR_RANGE` evaluates to `AUGMENT_SNR_RANGE_AUDIO` or `AUGMENT_SNR_RANGE_SEISMIC` based on `TRAIN_SENSOR`. No train.py change needed.

- [ ] **Step 2: No commit** — this task is a verification only.

---

## Task 4: Replace EpochShuffleSampler with WeightedRandomSampler in train.py

**Files:**
- Modify: `enforced-train/train.py:1-10, 121-147, 149-186`

`WeightedRandomSampler` requires per-sample weights derived from class counts. It replaces `EpochShuffleSampler` in the train DataLoader when `USE_WEIGHTED_SAMPLER=True`. The existing `_compute_class_weights` function already iterates `train_ds.samples` — reuse that logic to build per-sample weights.

- [ ] **Step 1: Add WeightedRandomSampler import**

In `enforced-train/train.py` line 6, change:
```python
from torch.utils.data import DataLoader, Sampler
```
to:
```python
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
```

- [ ] **Step 2: Add _build_weighted_sampler function**

After `_compute_class_weights` (after line 146), add:
```python
def _build_weighted_sampler(train_ds, config, device, strength: float = 1.0):
    """Build a WeightedRandomSampler from per-class inverse-frequency weights.

    strength=1.0 → fully class-balanced sampling.
    strength=0.0 → uniform sampling (equivalent to no sampler).
    Intermediate values interpolate between the two.
    """
    reverse_class_map = {v: k for k, v in config.CLASS_MAP.items()}
    class_weights = _compute_class_weights(train_ds, config, device, weight_cap=config.CLASS_WEIGHT_CAP)

    sample_weights = []
    for dataset, instance, _sensor_node, _run_id, _step_idx, label_str in train_ds.samples:
        if config.TRAINING_MODE == "detection":
            label_int = 0 if label_str == "background" else 1
        elif config.TRAINING_MODE == "category":
            label_int = reverse_class_map.get(label_str, 0)
        elif config.TRAINING_MODE == "instance":
            vehicle_type = config.DATASET_VEHICLE_MAP[dataset][instance][1]
            label_int = config.INSTANCE_TO_CLASS[vehicle_type]
        else:
            raise ValueError(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")

        w = class_weights[label_int].item()
        # Interpolate between uniform (1.0) and class-balanced (w) by strength
        sample_weights.append(1.0 + strength * (w - 1.0))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
```

- [ ] **Step 3: Replace train DataLoader construction in main()**

In `enforced-train/train.py`, replace lines 162–176:
```python
    epoch_sampler = EpochShuffleSampler(
        train_ds, base_seed=config.INSTANCE_SEED, num_epochs=config.EPOCHS
    )

    print(f"Total training samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=epoch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
```
with:
```python
    print(f"Total training samples: {len(train_ds)}")

    if getattr(config, "USE_WEIGHTED_SAMPLER", False):
        train_sampler = _build_weighted_sampler(
            train_ds, config, device,
            strength=getattr(config, "WEIGHTED_SAMPLER_STRENGTH", 1.0),
        )
        print(f"Using WeightedRandomSampler (strength={config.WEIGHTED_SAMPLER_STRENGTH})")
    else:
        train_sampler = EpochShuffleSampler(
            train_ds, base_seed=config.INSTANCE_SEED, num_epochs=config.EPOCHS
        )
        print("Using EpochShuffleSampler")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, "PIN_MEMORY", True),
        prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2),
        persistent_workers=getattr(config, "PERSISTENT_WORKERS", True),
    )
```

- [ ] **Step 4: Update val DataLoader with new kwargs**

Replace lines 178–186:
```python
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
```
with:
```python
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=getattr(config, "PIN_MEMORY", True),
        prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2),
        persistent_workers=getattr(config, "PERSISTENT_WORKERS", True),
    )
```

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "train: WeightedRandomSampler with configurable strength, updated DataLoader kwargs"
```

---

## Task 5: Add torch.compile and AMP bfloat16 to train.py

**Files:**
- Modify: `enforced-train/train.py:42-72, 204-218`

`torch.compile` wraps the model after initialization. AMP autocast wraps the forward pass in `train_one_epoch` and `evaluate`.

- [ ] **Step 1: Add torch.compile after model is moved to device**

In `enforced-train/train.py`, after line 208 (`.to(device)`):
```python
    model = build_model(
        input_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        config=config,
    ).to(device)

    if getattr(config, "TORCH_COMPILE", False):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
```

- [ ] **Step 2: Wrap forward pass in train_one_epoch with AMP autocast**

In `enforced-train/train.py`, modify `train_one_epoch` to accept an `amp_enabled` and `amp_dtype` parameter and wrap the forward/backward:

Replace the function signature and the optimizer/forward block (lines 42–72):
```python
def train_one_epoch(model, loader, optimizer, criterion, device, config, epoch,
                    steps_per_epoch=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    amp_enabled = getattr(config, "AMP_ENABLED", False) and device.type == "cuda"
    amp_dtype = getattr(torch, getattr(config, "AMP_DTYPE", "bfloat16"))

    for batch_idx, (x, y, dataset_names) in enumerate(loader):
        if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        if getattr(config, "AUGMENT_SNR", False):
            current_snr_range = getattr(config, "AUGMENT_SNR_RANGE", (10, 30))
            x = augment_batch(x, snr_range=current_snr_range)

        x = preprocess_for_training(x, config=config)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples
```

- [ ] **Step 3: Wrap evaluate forward pass with AMP autocast**

Replace the `evaluate` function (lines 75–118):
```python
def evaluate(model, loader, criterion, device, config, steps_per_epoch=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    amp_enabled = getattr(config, "AMP_ENABLED", False) and device.type == "cuda"
    amp_dtype = getattr(torch, getattr(config, "AMP_DTYPE", "bfloat16"))

    with torch.inference_mode():
        for batch_idx, (x, y, dataset_names) in enumerate(loader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            x, y = x.to(device), y.to(device)

            x = preprocess_for_training(x, config=config)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    total_samples = len(all_labels)
    avg_loss = total_loss / total_samples

    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    target_labels = list(range(config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    return avg_loss, accuracy, precision, recall, f1, per_class_acc
```

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "train: torch.compile and bfloat16 AMP autocast, steps_per_epoch enforcement"
```

---

## Task 6: Add MCC computation and switch checkpoint metric

**Files:**
- Modify: `enforced-train/train.py:1-12, 75-118, 240-355`

MCC requires `matthews_corrcoef` from scikit-learn. The training loop currently checkpoints on `val_f1`. Switch to `CHECKPOINT_METRIC` (defaulting to `"mcc"`) with MCC computed in `evaluate()`. `val_f1` must still be logged and saved in `meta.pt` for `evaluate_best_ensemble()`.

- [ ] **Step 1: Add matthews_corrcoef import**

In `enforced-train/train.py`, change the sklearn imports block:
```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
```
to:
```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)
```

- [ ] **Step 2: Add MCC to evaluate() return value**

In the `evaluate` function (updated in Task 5), add MCC computation before the return and update the return signature:

After the `per_class_acc` computation, add:
```python
    mcc = matthews_corrcoef(all_labels, all_preds)
```

Change the return from:
```python
    return avg_loss, accuracy, precision, recall, f1, per_class_acc
```
to:
```python
    return avg_loss, accuracy, precision, recall, f1, mcc, per_class_acc
```

- [ ] **Step 3: Update all callers of evaluate() to unpack mcc**

There are two call sites in `main()`:

The dummy-pass call (lines 212–217) does not call `evaluate()` — no change needed there.

The training loop call (around line 291):
```python
        val_loss, val_acc, val_prec, val_rec, val_f1, per_class_acc = evaluate(
            model, val_loader, criterion, device, config
        )
```
Replace with:
```python
        val_loss, val_acc, val_prec, val_rec, val_f1, val_mcc, per_class_acc = evaluate(
            model, val_loader, criterion, device, config,
            steps_per_epoch=getattr(config, "VAL_STEPS_PER_EPOCH", None),
        )
```

Also update the train call to pass steps:
```python
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch,
            steps_per_epoch=getattr(config, "TRAIN_STEPS_PER_EPOCH", None),
        )
```

- [ ] **Step 4: Add MCC to CSV headers and row**

In the CSV headers block (around line 254), add `"Val_MCC"` after `"Val_F1"`:
```python
    headers = [
        "Epoch",
        "Train_Loss",
        "Train_Acc",
        "Val_Loss",
        "Val_Acc",
        "Val_Precision",
        "Val_Recall",
        "Val_F1",
        "Val_MCC",
    ] + class_names
```

In the row_data block (around line 306), add `f"{val_mcc:.4f}"` after the F1 entry:
```python
            row_data = [
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{val_prec:.4f}",
                f"{val_rec:.4f}",
                f"{val_f1:.4f}",
                f"{val_mcc:.4f}",
            ] + class_values
```

- [ ] **Step 5: Add mcc to metrics_dict and wire CHECKPOINT_METRIC**

In the `metrics_dict` block (around line 318), add MCC:
```python
        metrics_dict = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_mcc": val_mcc,
        }
```

Change `target_metric_name` to read from config:
```python
    target_metric_name = getattr(config, "CHECKPOINT_METRIC", "val_f1")
    # Map "mcc" shorthand to the dict key
    if target_metric_name == "mcc":
        target_metric_name = "val_mcc"
```

- [ ] **Step 6: Wire EARLY_STOP_PATIENCE from config**

Replace hardcoded `EARLY_STOP_PATIENCE = 8` (line 281) with:
```python
    EARLY_STOP_PATIENCE = getattr(config, "EARLY_STOP_PATIENCE", 8)
```

- [ ] **Step 7: Wire CLASS_WEIGHT_CAP from config**

In `main()`, the call to `_compute_class_weights` (around line 223):
```python
        weights = _compute_class_weights(train_ds, config, device)
```
Replace with:
```python
        weights = _compute_class_weights(
            train_ds, config, device,
            weight_cap=getattr(config, "CLASS_WEIGHT_CAP", 10.0),
        )
```

- [ ] **Step 8: Commit**

```bash
git add train.py
git commit -m "train: MCC metric, configurable checkpoint metric and early-stop patience, CLASS_WEIGHT_CAP wired"
```

---

## Task 7: End-to-end smoke test

**Files:**
- Read: `enforced-train/run_pipeline.sh`

Before running all 36 models, run a single fast sanity check to confirm the config and train loop work without error.

- [ ] **Step 1: Run a single fast training run**

From `enforced-train/`:
```bash
TRAINING_MODE=detection TRAIN_SENSOR=audio MODEL_NAME=DetectionCNN \
    python train.py
```
Expected: training starts, prints device, prints WeightedRandomSampler or EpochShuffleSampler, prints class weights, runs epoch 1. Confirm no import errors, no shape errors in the forward pass, MCC appears in the CSV after epoch 1.

- [ ] **Step 2: Confirm metrics.csv contains Val_MCC column**

```bash
head -2 saved_models/detection/audio/DetectionCNN/*/metrics.csv
```
Expected: header row includes `Val_MCC`, data row shows a numeric value between -1 and 1.

- [ ] **Step 3: Confirm cache busted and rebuilding**

```bash
ls saved_models/cache/
```
Expected: new cache files with different hash suffixes than previous runs (due to `SAMPLE_SECONDS=2`). Old cache files with `SAMPLE_SECONDS=1` hash should not be loaded.

- [ ] **Step 4: Kill the smoke test run and launch full pipeline**

```bash
# Kill the smoke test if still running (Ctrl+C or kill PID)
# Then launch full pipeline:
bash run_pipeline.sh
```
Select all modes, all models, all sensors when prompted.

---

## Task 8: Verify improved results after full training run

**Files:**
- Read: `enforced-train/saved_models/master_evaluation_results.csv` (after pipeline completes)

- [ ] **Step 1: Check master_evaluation_results.csv for MCC improvements**

After `run_pipeline.sh` completes and `eval.py` + `aggregate_results.py` run, check:
```bash
python -c "
import pandas as pd
df = pd.read_csv('saved_models/master_evaluation_results.csv')
print(df.sort_values('MCC', ascending=False).to_string())
"
```
Expected: seismic InceptionTime and DetectionCNN show MCC > prior baselines (detection: 0.60, category: 0.35–0.45, instance: 0.19–0.39). Audio instance models show reduced per-class accuracy spread.

- [ ] **Step 2: Flag any seismic 1D model surprises**

Check whether any of BiGRU, ClassificationLSTM, WaveformClassificationCNN on seismic achieve MCC > 0.1 in any mode. If so, they become candidates for the ensemble — note them for the selection step.

- [ ] **Step 3: Select final 6 models for production pipeline**

Run `evaluate_best_ensemble()` (already implemented in `eval.py`) for each mode. It automatically selects the best 1 model per sensor per mode by `val_f1` from `meta.pt` files and reports the fused ensemble metrics.

Review: any model selected with MCC < 0.3 warrants manual inspection before production deployment.
