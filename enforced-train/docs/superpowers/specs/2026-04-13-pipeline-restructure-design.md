# Pipeline Restructure Design

**Date:** 2026-04-13  
**Status:** Approved

---

## Context

The `enforced-train` pipeline currently trains seismic and audio sensors together in a single loop and evaluates them via a same-architecture fused ensemble. Two underperforming models (MiniRocket, TCN) consume disproportionate training time. The ensemble logic couples models by architecture rather than selecting the best performer per sensor. This redesign:

1. Removes MiniRocket and TCN from the registry
2. Keeps interactive mode/model/sensor selection but parallelizes seismic and audio training within each mode
3. Evaluates each sensor independently, then produces a "best ensemble" that pairs the top seismic model with the top audio model (by val F1) for each pipeline mode

---

## Files Changed

| File | Change Type |
|------|-------------|
| `models.py` | Remove MiniRocket + TCN classes and registry entries |
| `config.py` | Remove MiniRocket/TCN from SHAPE_MAP; remove TCN_CHANNELS param |
| `train.py` | Remove `fit_extractor()` call (MiniRocket-only) |
| `run_pipeline.sh` | Parallelize seismic + audio jobs within each mode |
| `eval.py` | Remove `evaluate_fused()`; add `evaluate_best_ensemble()` |
| `aggregate_results.py` | Parse new `best_ensemble/` report path; add ensemble rows to master CSV |

---

## Section 1: Model Roster (`models.py` + `config.py`)

### Removed
- `_TemporalBlock` helper class
- `TCN` class
- `IterativeMiniRocket` class
- Both removed from `MODEL_REGISTRY` dict
- Both removed from `SHAPE_MAP` in `config.py`
- `TCN_CHANNELS` config param removed from `config.py`

### Remaining Registry (6 models)
| Key | Class | Input |
|-----|-------|-------|
| `detection_cnn` | DetectionCNN | 2D Mel spectrogram |
| `classification_cnn` | ClassificationCNN | 2D Mel spectrogram |
| `waveform_cnn` | WaveformClassificationCNN | 1D raw waveform |
| `classification_lstm` | ClassificationLSTM | 1D raw waveform |
| `inception_time` | InceptionTime | 1D raw waveform |
| `bigru` | BiGRU | 1D raw waveform |

No routing changes needed — `USE_MEL` flag derived from `SHAPE_MAP` still controls preprocessing for 2D vs 1D models.

### `train.py` cleanup
Remove the MiniRocket-only `fit_extractor()` call block (currently gated by `isinstance(model, IterativeMiniRocket)`). No other train loop changes.

---

## Section 2: Training Pipeline (`run_pipeline.sh`)

### Interactive Selection (preserved)
- Mode selection: `detection`, `category`, `instance`, or `all`
- Model selection: any subset of the 6 remaining models, or `all`
- Sensor selection: `audio`, `seismic`, or `all`
- Confirmation checkpoint before training begins

### Parallel Execution (new)
Within each mode, the selected models for seismic and audio are launched in parallel background processes:

```bash
for mode in "${selected_modes[@]}"; do
    # Launch seismic jobs (background)
    if [[ " ${selected_sensors[*]} " =~ seismic ]]; then
        for model in "${selected_models[@]}"; do
            RUN_ID=$(date +%s%N) TRAINING_MODE=$mode TRAIN_SENSOR=seismic \
                MODEL_NAME=$model poetry run python train.py &
            sleep 1
        done
    fi

    # Launch audio jobs (background, concurrent with seismic)
    if [[ " ${selected_sensors[*]} " =~ audio ]]; then
        for model in "${selected_models[@]}"; do
            RUN_ID=$(date +%s%N) TRAINING_MODE=$mode TRAIN_SENSOR=audio \
                MODEL_NAME=$model poetry run python train.py &
            sleep 1
        done
    fi

    wait  # Block until all jobs for this mode complete
done
```

Modes remain **sequential** to avoid GPU contention across 12+ simultaneous jobs. Within a mode, seismic and audio batches run concurrently.

### Post-training (unchanged entry points)
```bash
poetry run python eval.py
poetry run python aggregate_results.py
```

---

## Section 3: Evaluation (`eval.py`)

### Per-Sensor Eval (preserved)
`evaluate_directory()` runs unchanged for every saved model run, producing:
- `evaluation_report.txt`
- `conf_matrix.png`

### Removed
`evaluate_fused()` — the same-architecture cross-sensor fused eval is removed.

### New: `evaluate_best_ensemble(mode)`

```
1. Scan saved_models/{mode}/seismic/**/meta.pt
   → Load val_f1 from each; find the run_dir with highest val_f1
   → Call that the "best seismic model"

2. Scan saved_models/{mode}/audio/**/meta.pt
   → Same; find "best audio model"

3. If both found:
   a. Reconstruct each model from hyperparameters.json + best_model.pth
   b. Build separate test DataLoaders: one configured for the seismic model's sensor/sample-rate, one for audio — they cannot share a loader since preprocessing differs
   c. Run inference independently through each model on the test set
   d. Compute weighted softmax average:
        w_s = val_f1_seismic / (val_f1_seismic + val_f1_audio)
        w_a = val_f1_audio   / (val_f1_seismic + val_f1_audio)
        fused_probs = w_s * softmax(logits_seismic) + w_a * softmax(logits_audio)
        pred = argmax(fused_probs)
   e. Compute: accuracy, macro F1, MCC, ROC-AUC, mean latency per sample, FAR (detection only)
   f. Write evaluation_report.txt to:
        saved_models/{mode}/best_ensemble/evaluation_report.txt
      Write conf_matrix.png to:
        saved_models/{mode}/best_ensemble/conf_matrix.png
      Include in report header: which seismic model/run_id was selected and which audio model/run_id

4. If only one sensor's models exist, skip ensemble (log a warning).
```

`eval.py` main block calls `evaluate_best_ensemble(mode)` once per mode after all per-sensor evals are done.

---

## Section 4: Aggregation (`aggregate_results.py`)

### New Report Source
Add `saved_models/{mode}/best_ensemble/evaluation_report.txt` to the scan paths.

Tag these rows with `Sensor = "best_ensemble"` in the master CSV. `Model` column populated from the report header (e.g., `"inception_time + classification_cnn"`).

### CSV Schema (unchanged)
Columns: `Mode, Sensor, Model, Accuracy, F1-Score, Precision, Recall, MCC, ROC-AUC, Latency_ms, FAR, Timestamp`

### Console Leaderboard
Ensemble rows appear at the bottom of each mode group, visually separated, ranked by F1-Score.

---

## Verification

1. **Smoke test model removal**: `python -c "from models import MODEL_REGISTRY; print(list(MODEL_REGISTRY))"` — should list exactly 6 models with no MiniRocket/TCN.
2. **Smoke test config**: `python -c "from config import SHAPE_MAP; print(SHAPE_MAP)"` — MiniRocket/TCN keys absent.
3. **Single train run**: Train one model on one sensor for 2 epochs to confirm `fit_extractor` removal doesn't break the training loop.
4. **Full pipeline dry run**: Run `run_pipeline.sh` selecting 1 mode, 2 models, both sensors — confirm jobs launch in parallel, `wait` blocks correctly.
5. **Ensemble eval**: After training, confirm `saved_models/{mode}/best_ensemble/evaluation_report.txt` is created with correct metrics and identifies the source models.
6. **Aggregate**: Confirm `master_evaluation_results.csv` includes `best_ensemble` rows alongside per-sensor rows.
