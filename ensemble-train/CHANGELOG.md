# Pipeline Improvement Changelog

## Summary of Changes

All files have been cleaned up for consistent style, reduced duplication, and improved
training performance. Changes are grouped by the priority areas you identified.

---

## 1. General Code Cleanup (All Files)

### models.py
- **Removed unused imports** (`time`, `numpy`, `get_minirocket_features`)
- **Replaced scattered conv/relu/pool chains with `nn.Sequential` blocks** — `self.features` and `self.classifier` make the architecture scannable at a glance
- **Added `_init_weights` helper** — Kaiming init for conv layers, Xavier for linear. Applied via `self.apply(_init_weights)` in every model constructor
- **Fixed LSTM dropout warning** — PyTorch warns when `dropout > 0` and `num_layers == 1`. Now conditionally sets `dropout=0.0` when `LAYERS == 1`

### train.py
- **Extracted `build_batch_sigma()`** — the sigma-mapping logic was copy-pasted in `train_one_epoch`, `evaluate`, and the dummy forward pass. Now a single shared function
- **Extracted `build_class_headers()`** — the CSV header construction was a long if/elif block; now a clean helper
- **Decorated `evaluate()` with `@torch.inference_mode()`** — removes the manual `with` block
- **Added current LR to epoch printout and CSV** — easier to see scheduler behavior in logs

### eval.py
- **Extracted `build_batch_sigma()`** (same function as train.py — could be moved to a shared `utils.py` in a future refactor)
- **Extracted `build_axis_labels()`** — confusion matrix label generation was duplicated
- **Extracted `compute_auc()`** — the try/except AUC logic is now a clean function
- **Sorted run directories** for deterministic evaluation order

### dataset.py
- **Renamed internal methods** for clarity: `_get_tables` → `_discover_tables`, `_get_table_max_time` → `_compute_time_bounds`, `_align_max_time` → `_align_sensor_groups`, `_get_samples` → `_build_samples`
- **Extracted `_resolve_label()`** — the label string → int conversion was inline in `__getitem__`
- **Extracted `_should_synthesize()`** and `_make_synthetic_sample()`** — synthetic generation logic was a long nested conditional
- **Extracted `_balance_backgrounds()`** — oversampling logic separated from sample building
- **Removed the confusing `self.noise_floor = 0.01` vs `self.noise_floors = {}`** — now only `self.noise_floors = {}` exists, and the synthetic generator handles the empty-dict fallback

### data_generator.py
- **Replaced `torch.max(...)[0]` with `.amax()`** — cleaner, same result
- **Used `.sqrt()` method** instead of `torch.sqrt()` for readability

### db_utils.py
- **Simplified `db_connect`** — cursor created inline from conn
- **Compact SQL formatting** — same logic, fewer lines

### aggregate_results.py
- **Used `map()` for parsing** — replaced explicit loop
- **Local `fmt()` helper** for console output formatting

### run_pipeline.sh
- **Reduced visual noise** — same logic, tighter prompts

---

## 2. LR Scheduling, Early Stopping, Batch Norm

### models.py
- **Added `nn.BatchNorm2d` / `nn.BatchNorm1d`** after every convolution layer in all four CNN/LSTM models. This is probably the single highest-impact change for training stability — batch norm reduces internal covariate shift and allows higher learning rates
- **Fixed `ClassificationCNN` activation pattern** — the original alternated `tanh` → `relu` → `tanh` → `relu`. Tanh saturates to ±1 and creates vanishing gradient problems. Now all activations are `ReLU`

### train.py
- **Added `ReduceLROnPlateau` scheduler** — monitors val F1 (or val_loss if that's your metric), halves LR after 3 plateau epochs. This is conservative and safe for all model types
- **Added `EarlyStopping` class** — configurable patience (default 8 epochs), tracks the best model metric and stops training when no improvement is seen. Controlled by new `EARLY_STOP_PATIENCE` config variable
- **Added gradient clipping** (`nn.utils.clip_grad_norm_`, default max_norm=1.0) — particularly important for the LSTM, which can get exploding gradients on noisy seismic data

### config.py
- **`EPOCHS` increased from 5 → 30** — with early stopping, this is a ceiling not a target. The old 5 epochs was almost certainly not enough for convergence
- **Added `EARLY_STOP_PATIENCE = 8`**
- **Added `GRAD_CLIP = 1.0`**

---

## 3. Spectral / Model Fixes

### config.py
- **`N_FFT` and `HOP_LENGTH` are now computed dynamically** based on `REF_SAMPLE_RATE * SAMPLE_SECONDS`:
  - `N_FFT = min(1024, 2^floor(log2(signal_length)))`
  - `HOP_LENGTH = max(1, N_FFT // 4)`
  
  **Why this matters**: With `TRAIN_SENSORS = ["seismic"]`, `REF_SAMPLE_RATE = 200`, and `SAMPLE_SECONDS = 1`, the signal is only 200 samples long. The old hardcoded `N_FFT = 1024` was 5× the signal length — the STFT was mostly zero-padding, producing a single degenerate time frame. The 2D models (DetectionCNN, ClassificationCNN) were getting spectrograms with almost no temporal structure. Now `N_FFT = 128` and `HOP_LENGTH = 32`, giving ~6 time frames — still compact, but enough for the 2D convolutions to learn from.

- **Added `CLASS_WEIGHTS` for instance mode** — the original left `CLASS_WEIGHTS = []` for instance training, which silently fell back to unweighted CrossEntropyLoss. This meant rare vehicle instances were drowned out. Now initialized to uniform `[1.0] * NUM_CLASSES` (you can tune these with actual per-instance sample counts later).

---

## 4. Config Refactoring (Light)

### config.py
- **Switched model hyperparameter blocks to `elif`** — the original used independent `if` blocks, meaning every block was evaluated even though only one matched. `elif` is clearer about mutual exclusivity and avoids accidentally defining variables for the wrong model
- **Removed dead `table_max_time` and `split_idx` dicts** from dataset (they were initialized but unused)
- **Kept interactive `input()` calls** — full refactoring into a dataclass or argparse was ranked lowest priority; the current structure works for your pipeline

---

## What to Try Next

1. **Run the full pipeline** and compare metrics against your previous results. The batch norm + spectral fix should show immediate improvement on the 2D models.

2. **Tune `CLASS_WEIGHTS` for instance mode** — run one epoch, count samples per class, and set inverse-frequency weights.

3. **Consider a shared `utils.py`** for `build_batch_sigma()` and similar helpers that are duplicated between train.py and eval.py.

4. **If seismic-only 2D results are still weak**, consider whether mel spectrograms are the right representation at 200 Hz — a plain STFT or even raw waveform (1D models) may be more appropriate for sub-audio signals.
