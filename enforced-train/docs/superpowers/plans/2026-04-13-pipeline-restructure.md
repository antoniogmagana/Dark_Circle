# Pipeline Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove MiniRocket and TCN from the training pipeline, parallelize seismic/audio training within each mode, and replace same-architecture fused eval with a best-of-each-sensor ensemble on the test set.

**Architecture:** Six files are touched in sequence: models.py and config.py (model removal), train.py (MiniRocket cleanup), run_pipeline.sh (parallel training), eval.py (new ensemble function replacing evaluate_fused), aggregate_results.py (parse new best_ensemble path). No new files are created.

**Tech Stack:** PyTorch, scikit-learn, seaborn/matplotlib, pandas, bash, poetry

---

## File Map

| File | Change |
|------|--------|
| `enforced-train/models.py` | Delete `_TemporalBlock`, `TCN`, `IterativeMiniRocket` classes; remove from `MODEL_REGISTRY` |
| `enforced-train/config.py` | Remove `IterativeMiniRocket`/`TCN` elif blocks and `SHAPE_MAP` entries |
| `enforced-train/train.py` | Remove `fit_extractor()` call block in dummy-pass section |
| `enforced-train/run_pipeline.sh` | Update model menu (6 models), parallelize training loop within each mode |
| `enforced-train/eval.py` | Delete `evaluate_fused()`; add `evaluate_best_ensemble(mode_dir)`; update `main()` |
| `enforced-train/aggregate_results.py` | Add `best_ensemble` path to scan; tag rows; separate leaderboard section |

---

## Task 1: Remove TCN and MiniRocket from `models.py`

**Files:**
- Modify: `enforced-train/models.py:306-499`

- [ ] **Step 1: Delete `_TemporalBlock` and `TCN` classes**

Remove lines 306–369 (the `_TemporalBlock` class and the `TCN` class) from `models.py`.

The section to delete starts with:
```python
class _TemporalBlock(nn.Module):
    """Dilated causal residual block used by TCN."""
```
and ends with the closing brace of `TCN.get_optimizer()`.

- [ ] **Step 2: Delete `IterativeMiniRocket` class**

Remove lines 432–483 (the `IterativeMiniRocket` class including its docstring and all methods) from `models.py`.

The section to delete starts with:
```python
class IterativeMiniRocket(nn.Module):
    """
    End-to-End PyTorch MiniRocket.
```
and ends with the closing brace of `IterativeMiniRocket.get_optimizer()`.

- [ ] **Step 3: Remove the tsai import**

Delete this line at the top of `models.py`:
```python
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
```

- [ ] **Step 4: Update `MODEL_REGISTRY` to 6 models**

Replace the current `MODEL_REGISTRY` dict:
```python
MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "IterativeMiniRocket": IterativeMiniRocket,
    "InceptionTime": InceptionTime,
    "TCN": TCN,
    "BiGRU": BiGRU,
}
```
with:
```python
MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "InceptionTime": InceptionTime,
    "BiGRU": BiGRU,
}
```

- [ ] **Step 5: Verify the registry smoke test passes**

Run from `enforced-train/`:
```bash
poetry run python -c "from models import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY))"
```
Expected output:
```
['BiGRU', 'ClassificationCNN', 'ClassificationLSTM', 'DetectionCNN', 'InceptionTime', 'WaveformClassificationCNN']
```

- [ ] **Step 6: Commit**

```bash
cd enforced-train
git add models.py
git commit -m "feat: remove TCN and MiniRocket from model registry"
```

---

## Task 2: Remove TCN and MiniRocket config from `config.py`

**Files:**
- Modify: `enforced-train/config.py:311-378`

- [ ] **Step 1: Delete the `IterativeMiniRocket` elif block**

In `config.py`, remove this entire block (lines ~311–316):
```python
# --- miniROCKET ---
elif MODEL_NAME == "IterativeMiniRocket":
    LEARNING_RATE = 1e-3
    DROPOUT = 0.3
    MINIROCKET_FEATURES = 1000
    # The tsai extractor defaults to 10,000 kernels automatically
```

- [ ] **Step 2: Delete the `TCN` elif block**

Remove this entire block (lines ~336–343):
```python
# --- TCN ---
elif MODEL_NAME == "TCN":
    LEARNING_RATE = 1e-3
    TCN_CHANNELS = 64  # filters per dilated conv level
    TCN_KERNEL_SIZE = 7  # kernel size for all dilated convolutions
    TCN_LEVELS = 4  # dilation = 1,2,4,8 → receptive field ≈ 91 samples
    HIDDEN = 128
    DROPOUT = 0.2
```

- [ ] **Step 3: Remove TCN and MiniRocket from `SHAPE_MAP`**

Replace the current `SHAPE_MAP`:
```python
SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "IterativeMiniRocket": "1D",
    "InceptionTime": "1D",
    "TCN": "1D",
    "BiGRU": "1D",
}
```
with:
```python
SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "InceptionTime": "1D",
    "BiGRU": "1D",
}
```

- [ ] **Step 4: Verify SHAPE_MAP smoke test**

```bash
poetry run python -c "from config import SHAPE_MAP; print(sorted(SHAPE_MAP))"
```
Expected output (MODEL_NAME prompt will appear — enter any valid name like `DetectionCNN`):
```
['BiGRU', 'ClassificationCNN', 'ClassificationLSTM', 'DetectionCNN', 'InceptionTime', 'WaveformClassificationCNN']
```

- [ ] **Step 5: Commit**

```bash
git add config.py
git commit -m "feat: remove TCN and MiniRocket config hyperparameter blocks"
```

---

## Task 3: Remove `fit_extractor()` call from `train.py`

**Files:**
- Modify: `enforced-train/train.py:183-195`

- [ ] **Step 1: Simplify the dummy-pass block**

In `train.py`, find the dummy-pass section (lines ~183–195):
```python
    print("Performing dummy pass to initialize Lazy modules...")
    model.eval()
    with torch.no_grad():
        for x_dummy, _, ds_names in train_loader:
            x_dummy = x_dummy.to(device)

            x_dummy = preprocess_for_training(x_dummy, config=config)

            if hasattr(model, "fit_extractor"):
                model.fit_extractor(x_dummy[:32])
                model(x_dummy[:32])
            else:
                model(x_dummy)
            break
```

Replace with:
```python
    print("Performing dummy pass to initialize Lazy modules...")
    model.eval()
    with torch.no_grad():
        for x_dummy, _, ds_names in train_loader:
            x_dummy = x_dummy.to(device)
            x_dummy = preprocess_for_training(x_dummy, config=config)
            model(x_dummy)
            break
```

- [ ] **Step 2: Verify train.py imports cleanly**

```bash
poetry run python -c "import train; print('train.py imports OK')"
```
Expected: `train.py imports OK` (TRAINING_MODE and MODEL_NAME prompts will appear — enter valid values like `detection` and `DetectionCNN`).

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: remove MiniRocket fit_extractor call from training loop"
```

---

## Task 4: Update `run_pipeline.sh` — model menu and parallel training

**Files:**
- Modify: `enforced-train/run_pipeline.sh:29-132`

- [ ] **Step 1: Update the model selection menu (section 2)**

Replace the entire model selection section (lines 29–61):
```bash
# --- 2. Interactive Model Selection ---
echo ""
echo "============================================================"
echo " SELECT MODELS"
echo "============================================================"
echo "Enter the numbers of the models you want to train (space-separated)."
echo "Or type 'all' to select everything."
echo ""
echo "  1) DetectionCNN"
echo "  2) ClassificationCNN"
echo "  3) WaveformClassificationCNN"
echo "  4) ClassificationLSTM"
echo "  5) IterativeMiniRocket"
echo "  6) InceptionTime"
echo "  7) TCN"
echo "  8) BiGRU"
echo "  9) ALL MODELS"
echo ""
read -p "Models > " model_input

selected_models=()
if [[ "$model_input" == *"9"* ]] || [[ "${model_input,,}" == *"all"* ]]; then
    selected_models=("DetectionCNN" "ClassificationCNN" "WaveformClassificationCNN" "ClassificationLSTM" "IterativeMiniRocket" "InceptionTime" "TCN" "BiGRU")
else
    [[ "$model_input" == *"1"* ]] && selected_models+=("DetectionCNN")
    [[ "$model_input" == *"2"* ]] && selected_models+=("ClassificationCNN")
    [[ "$model_input" == *"3"* ]] && selected_models+=("WaveformClassificationCNN")
    [[ "$model_input" == *"4"* ]] && selected_models+=("ClassificationLSTM")
    [[ "$model_input" == *"5"* ]] && selected_models+=("IterativeMiniRocket")
    [[ "$model_input" == *"6"* ]] && selected_models+=("InceptionTime")
    [[ "$model_input" == *"7"* ]] && selected_models+=("TCN")
    [[ "$model_input" == *"8"* ]] && selected_models+=("BiGRU")
fi
```

with:
```bash
# --- 2. Interactive Model Selection ---
echo ""
echo "============================================================"
echo " SELECT MODELS"
echo "============================================================"
echo "Enter the numbers of the models you want to train (space-separated)."
echo "Or type 'all' to select everything."
echo ""
echo "  1) DetectionCNN"
echo "  2) ClassificationCNN"
echo "  3) WaveformClassificationCNN"
echo "  4) ClassificationLSTM"
echo "  5) InceptionTime"
echo "  6) BiGRU"
echo "  7) ALL MODELS"
echo ""
read -p "Models > " model_input

selected_models=()
if [[ "$model_input" == *"7"* ]] || [[ "${model_input,,}" == *"all"* ]]; then
    selected_models=("DetectionCNN" "ClassificationCNN" "WaveformClassificationCNN" "ClassificationLSTM" "InceptionTime" "BiGRU")
else
    [[ "$model_input" == *"1"* ]] && selected_models+=("DetectionCNN")
    [[ "$model_input" == *"2"* ]] && selected_models+=("ClassificationCNN")
    [[ "$model_input" == *"3"* ]] && selected_models+=("WaveformClassificationCNN")
    [[ "$model_input" == *"4"* ]] && selected_models+=("ClassificationLSTM")
    [[ "$model_input" == *"5"* ]] && selected_models+=("InceptionTime")
    [[ "$model_input" == *"6"* ]] && selected_models+=("BiGRU")
fi
```

- [ ] **Step 2: Replace the sequential training loop with a parallel loop (section 5)**

Replace the entire training loop section (lines 114–132):
```bash
# --- 5. Unified Model Training Pipeline ---
echo ""
for mode in "${selected_modes[@]}"; do
    for sensor in "${selected_sensors[@]}"; do
        for model in "${selected_models[@]}"; do
            CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)

            echo "============================================================"
            echo "STARTING RUN -> MODE: $mode | SENSOR: $sensor | MODEL: $model | RUN_ID: $CURRENT_RUN_ID"
            echo "============================================================"

            # Run training loop
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode TRAIN_SENSOR=$sensor MODEL_NAME=$model poetry run python train.py

            # Force a 1-second delay to guarantee unique RUN_IDs
            sleep 1
        done
    done
done
```

with:
```bash
# --- 5. Unified Model Training Pipeline (parallel by sensor within each mode) ---
echo ""
for mode in "${selected_modes[@]}"; do
    echo "============================================================"
    echo "MODE: $mode — launching sensor jobs in parallel"
    echo "============================================================"

    # Launch all seismic jobs in the background
    if [[ " ${selected_sensors[*]} " =~ " seismic " ]]; then
        for model in "${selected_models[@]}"; do
            CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
            echo "  [BG] seismic | $model | $CURRENT_RUN_ID"
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode TRAIN_SENSOR=seismic MODEL_NAME=$model \
                poetry run python train.py &
            sleep 1
        done
    fi

    # Launch all audio jobs in the background (concurrent with seismic)
    if [[ " ${selected_sensors[*]} " =~ " audio " ]]; then
        for model in "${selected_models[@]}"; do
            CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
            echo "  [BG] audio   | $model | $CURRENT_RUN_ID"
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode TRAIN_SENSOR=audio MODEL_NAME=$model \
                poetry run python train.py &
            sleep 1
        done
    fi

    echo "  Waiting for all $mode jobs to finish..."
    wait
    echo "  $mode complete."
done
```

- [ ] **Step 3: Remove `set -e` from the top of the script**

The current script has `set -e` on line 4, which causes the whole script to exit if any background job fails. Since we are now backgrounding jobs, a single failed training run would kill the script before `wait` collects all results. Remove that line:

Delete:
```bash
# Exit immediately if a pipeline fails entirely
set -e 
```

- [ ] **Step 4: Verify the script is valid bash**

```bash
bash -n enforced-train/run_pipeline.sh
```
Expected: no output (no syntax errors).

- [ ] **Step 5: Commit**

```bash
git add enforced-train/run_pipeline.sh
git commit -m "feat: update run_pipeline.sh — remove TCN/MiniRocket from menu, parallelize training by sensor"
```

---

## Task 5: Replace `evaluate_fused()` with `evaluate_best_ensemble()` in `eval.py`

**Files:**
- Modify: `enforced-train/eval.py:293-545`

- [ ] **Step 1: Delete the `evaluate_fused()` function**

Remove lines 293–526 (the entire `evaluate_fused` function body and its docstring). The function starts with:
```python
def evaluate_fused(mode_dir):
    """
    Fuse per-sensor models for one training mode via weighted average softmax.
```
and ends before `def main():`.

- [ ] **Step 2: Add `evaluate_best_ensemble(mode_dir)` in place of `evaluate_fused`**

Insert the following function at the same location (before `def main():`):

```python
def evaluate_best_ensemble(mode_dir):
    """
    For a given training mode directory, find the single best seismic model
    and the single best audio model (by val_f1 stored in meta.pt), then run
    a weighted softmax ensemble on the test set and write the result to
    saved_models/{mode}/best_ensemble/.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    def _find_best_run(sensor_dir):
        """Scan all meta.pt files under sensor_dir; return (run_dir, val_f1, model_name) for the best."""
        best_val_f1 = -1.0
        best_run_dir = None
        best_model_name = None
        if not sensor_dir.exists():
            return None, -1.0, None
        for meta_path in sensor_dir.rglob("meta.pt"):
            run_dir = meta_path.parent
            try:
                meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            val_f1 = meta.get("val_f1", 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_run_dir = run_dir
                best_model_name = meta.get("model_name", run_dir.parent.name)
        return best_run_dir, best_val_f1, best_model_name

    seismic_dir = mode_dir / "seismic"
    audio_dir = mode_dir / "audio"

    seismic_run_dir, seismic_f1, seismic_model_name = _find_best_run(seismic_dir)
    audio_run_dir, audio_f1, audio_model_name = _find_best_run(audio_dir)

    if seismic_run_dir is None or audio_run_dir is None:
        missing = []
        if seismic_run_dir is None:
            missing.append("seismic")
        if audio_run_dir is None:
            missing.append("audio")
        print(
            f"  [!] Best-ensemble skipped for {mode_dir.name}: "
            f"no trained models found for {', '.join(missing)}."
        )
        return

    ensemble_dir = mode_dir / "best_ensemble"
    report_path = ensemble_dir / "evaluation_report.txt"
    if report_path.exists():
        print(f"Skipping best_ensemble (exists): {ensemble_dir}")
        return

    print(
        f"\nBest-ensemble eval: mode={mode_dir.name} | "
        f"seismic={seismic_model_name} (f1={seismic_f1:.4f}) | "
        f"audio={audio_model_name} (f1={audio_f1:.4f})"
    )

    def _load_sensor_model(run_dir):
        """Load model + per-sensor config from a run directory."""
        json_path = run_dir / "hyperparameters.json"
        meta_path = run_dir / "meta.pt"
        model_path = run_dir / "best_model.pth"
        if not json_path.exists() or not meta_path.exists() or not model_path.exists():
            return None, None
        with open(json_path, "r") as f:
            cfg_dict = json.load(f)
        if "CLASS_MAP" in cfg_dict:
            cfg_dict["CLASS_MAP"] = {int(k): v for k, v in cfg_dict["CLASS_MAP"].items()}
        s_cfg = SimpleNamespace(**cfg_dict)
        s_cfg.DEVICE = device_str
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        s_cfg.USE_MEL = meta.get("use_mel", getattr(s_cfg, "USE_MEL", False))
        m = build_model(
            input_channels=s_cfg.IN_CHANNELS,
            num_classes=s_cfg.NUM_CLASSES,
            config=s_cfg,
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        m.load_state_dict(state_dict)
        m.eval()
        return m, s_cfg

    seismic_model, seismic_cfg = _load_sensor_model(seismic_run_dir)
    audio_model, audio_cfg = _load_sensor_model(audio_run_dir)

    if seismic_model is None or audio_model is None:
        print(f"  [!] Could not load one or both models for best_ensemble. Skipping.")
        return

    # Build separate test DataLoaders — preprocessing differs between sensors
    seismic_test_ds = VehicleDataset(split="test", config=seismic_cfg)
    audio_test_ds = VehicleDataset(split="test", config=audio_cfg)

    if len(seismic_test_ds) == 0 or len(audio_test_ds) == 0:
        print(f"  [!] Empty test dataset for best_ensemble. Skipping.")
        return

    seismic_loader = DataLoader(
        seismic_test_ds,
        batch_size=seismic_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=seismic_cfg.NUM_WORKERS,
    )
    audio_loader = DataLoader(
        audio_test_ds,
        batch_size=audio_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=audio_cfg.NUM_WORKERS,
    )

    # Infer on each sensor independently, collect per-sample probabilities
    def _run_inference(model, loader, cfg):
        all_probs = []
        all_labels = []
        with torch.inference_mode():
            for batch in loader:
                x, y = batch[0], batch[1]
                x = x.to(device)
                x = preprocess_for_training(x, config=cfg)
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu()
                all_probs.append(probs)
                all_labels.append(y)
        return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)

    # Use the config with the correct label space (same NUM_CLASSES for both)
    ref_cfg = seismic_cfg

    start_time = time.perf_counter()
    seismic_probs, seismic_labels = _run_inference(seismic_model, seismic_loader, seismic_cfg)
    audio_probs, audio_labels = _run_inference(audio_model, audio_loader, audio_cfg)
    end_time = time.perf_counter()

    # Sanity check: both loaders must yield the same labels in the same order
    if not torch.equal(seismic_labels, audio_labels):
        print(
            "  [!] Label mismatch between seismic and audio test sets. "
            "Ensemble results may be unreliable."
        )

    all_labels = seismic_labels.numpy()
    total_samples = len(all_labels)

    # Weighted softmax average
    w_s = seismic_f1 / (seismic_f1 + audio_f1) if (seismic_f1 + audio_f1) > 0 else 0.5
    w_a = audio_f1 / (seismic_f1 + audio_f1) if (seismic_f1 + audio_f1) > 0 else 0.5
    fused_probs = (w_s * seismic_probs + w_a * audio_probs).numpy()
    all_preds = fused_probs.argmax(axis=1)

    latency_ms = ((end_time - start_time) / total_samples) * 1000

    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    unique_classes = len(np.unique(all_labels))
    if unique_classes > 1:
        try:
            if ref_cfg.NUM_CLASSES == 2:
                auc = roc_auc_score(all_labels, fused_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, fused_probs, multi_class="ovr")
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    target_labels = list(range(ref_cfg.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)

    far = None
    if ref_cfg.TRAINING_MODE == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    ensemble_dir.mkdir(parents=True, exist_ok=True)

    combined_model_label = f"{seismic_model_name} + {audio_model_name}"

    with open(report_path, "w") as f:
        f.write(f"Run Directory: best_ensemble\n")
        f.write(
            f"Mode: {ref_cfg.TRAINING_MODE} | Model: {combined_model_label} "
            f"[best_ensemble]\n"
        )
        f.write(
            f"Seismic: {seismic_model_name} (run: {seismic_run_dir.name}, val_f1={seismic_f1:.4f})\n"
        )
        f.write(
            f"Audio:   {audio_model_name} (run: {audio_run_dir.name}, val_f1={audio_f1:.4f})\n"
        )
        f.write(f"Ensemble weights: seismic={w_s:.4f}, audio={w_a:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Latency: {latency_ms:.4f} ms/sample\n")
        if far is not None:
            f.write(f"False Alarm Rate: {far * 100:.3f}%\n")
        f.write("\nPer-Class Accuracy:\n")
        if ref_cfg.TRAINING_MODE == "detection":
            f.write(f"  Background (0): {per_class_acc[0]:.4f}\n")
            f.write(f"  Target (1): {per_class_acc[1]:.4f}\n")
        elif ref_cfg.TRAINING_MODE == "category":
            for k, v in ref_cfg.CLASS_MAP.items():
                if k < len(per_class_acc):
                    f.write(f"  {v} ({k}): {per_class_acc[k]:.4f}\n")
        elif ref_cfg.TRAINING_MODE == "instance":
            inv_map = {v: k for k, v in ref_cfg.INSTANCE_TO_CLASS.items()}
            for k in range(ref_cfg.NUM_CLASSES):
                if k < len(per_class_acc):
                    name = inv_map.get(k, f"Class_{k}")
                    f.write(f"  {name} ({k}): {per_class_acc[k]:.4f}\n")

    # Confusion matrix plot (reuse same style as evaluate_directory)
    axis_labels = []
    if ref_cfg.TRAINING_MODE == "detection":
        axis_labels = ["background", "target"]
    elif ref_cfg.TRAINING_MODE == "category":
        axis_labels = [ref_cfg.CLASS_MAP.get(i, str(i)) for i in range(ref_cfg.NUM_CLASSES)]
    elif ref_cfg.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in getattr(ref_cfg, "INSTANCE_TO_CLASS", {}).items()}
        axis_labels = [inv_map.get(i, str(i)) for i in range(ref_cfg.NUM_CLASSES)]
    else:
        axis_labels = [str(i) for i in range(ref_cfg.NUM_CLASSES)]

    fig_size = max(12, ref_cfg.NUM_CLASSES * 1.2)
    annot_size = max(18, min(26, int(240 / ref_cfg.NUM_CLASSES)))

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": annot_size, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
        xticklabels=axis_labels,
        yticklabels=axis_labels,
    )
    plt.title(
        f"Confusion Matrix: {combined_model_label} ({ref_cfg.TRAINING_MODE}) [Best Ensemble]",
        fontsize=22,
        pad=20,
    )
    plt.ylabel("True Label", fontsize=22, labelpad=14)
    plt.xlabel("Predicted Label", fontsize=22, labelpad=14)
    if ref_cfg.NUM_CLASSES > 5:
        plt.xticks(rotation=45, ha="right", fontsize=20)
    else:
        plt.xticks(rotation=0, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.gcf().axes[-1].tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(ensemble_dir / "conf_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Best-ensemble report saved to {report_path}")
```

- [ ] **Step 3: Update `main()` to call `evaluate_best_ensemble` instead of `evaluate_fused`**

Replace the current `main()` function:
```python
def main():
    base_dir = Path("saved_models")
    if not base_dir.exists():
        print("No saved_models directory found.")
        return

    # Per-sensor evaluation
    run_dirs = [p.parent for p in base_dir.rglob("best_model.pth")]
    for run_dir in run_dirs:
        evaluate_directory(run_dir)

    # Fused evaluation: one pass per training mode directory
    mode_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name != "cache"
    ]
    for mode_dir in mode_dirs:
        evaluate_fused(mode_dir)
```

with:
```python
def main():
    base_dir = Path("saved_models")
    if not base_dir.exists():
        print("No saved_models directory found.")
        return

    # Per-sensor evaluation — skip runs inside best_ensemble dirs
    run_dirs = [
        p.parent for p in base_dir.rglob("best_model.pth")
        if "best_ensemble" not in p.parts
    ]
    for run_dir in run_dirs:
        evaluate_directory(run_dir)

    # Best-ensemble evaluation: one pass per training mode directory
    mode_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in ("cache",)
    ]
    for mode_dir in mode_dirs:
        evaluate_best_ensemble(mode_dir)
```

- [ ] **Step 4: Verify eval.py imports cleanly**

```bash
poetry run python -c "import eval; print('eval.py imports OK')"
```
Expected: `eval.py imports OK`

- [ ] **Step 5: Commit**

```bash
git add eval.py
git commit -m "feat: replace evaluate_fused with evaluate_best_ensemble in eval.py"
```

---

## Task 6: Update `aggregate_results.py` to parse `best_ensemble` reports

**Files:**
- Modify: `enforced-train/aggregate_results.py:70-168`

- [ ] **Step 1: Update `_sensor_from_path()` to recognize `best_ensemble`**

Replace the current `_sensor_from_path` function:
```python
def _sensor_from_path(filepath):
    """
    Infer the sensor name from the report file path.
    Per-sensor:  saved_models/{mode}/{sensor}/{model}/{run_id}/report.txt
    Fused:       saved_models/{mode}/fused/{model}_{run_id}/report_fused.txt
    """
    parts = os.path.normpath(filepath).split(os.sep)
    # parts[-1] = filename, [-2] = run_id, [-3] = model, [-4] = sensor/fused
    if len(parts) >= 4:
        return parts[-4]
    return "unknown"
```

with:
```python
def _sensor_from_path(filepath):
    """
    Infer the sensor name from the report file path.
    Per-sensor:     saved_models/{mode}/{sensor}/{model}/{run_id}/report.txt → parts[-4] = sensor
    Best-ensemble:  saved_models/{mode}/best_ensemble/report.txt             → parts[-2] = 'best_ensemble'
    """
    parts = os.path.normpath(filepath).split(os.sep)
    # Check for best_ensemble (3-level deep: saved_models/{mode}/best_ensemble/report.txt)
    if len(parts) >= 2 and parts[-2] == "best_ensemble":
        return "best_ensemble"
    # Per-sensor: saved_models/{mode}/{sensor}/{model}/{run_id}/report.txt
    if len(parts) >= 4:
        return parts[-4]
    return "unknown"
```

- [ ] **Step 2: Add `best_ensemble` glob pattern to `main()`**

Replace the scan section in `main()`:
```python
    # Per-sensor: saved_models/{mode}/{sensor}/{model}/{run_id}/
    sensor_pattern = os.path.join(
        "saved_models", "*", "*", "*", "*", "evaluation_report.txt"
    )
    # Fused: saved_models/{mode}/fused/{model}_{run_id}/
    fused_pattern = os.path.join(
        "saved_models", "*", "fused", "*", "evaluation_report.txt"
    )
    report_files = glob.glob(sensor_pattern) + glob.glob(fused_pattern)
```

with:
```python
    # Per-sensor: saved_models/{mode}/{sensor}/{model}/{run_id}/
    sensor_pattern = os.path.join(
        "saved_models", "*", "*", "*", "*", "evaluation_report.txt"
    )
    # Best-ensemble: saved_models/{mode}/best_ensemble/
    ensemble_pattern = os.path.join(
        "saved_models", "*", "best_ensemble", "evaluation_report.txt"
    )
    report_files = glob.glob(sensor_pattern) + glob.glob(ensemble_pattern)
```

- [ ] **Step 3: Add a visual separator for ensemble rows in the console leaderboard**

In the console output loop in `main()`, add a separator before ensemble rows. Replace:
```python
        print(f"{row['Model']:<30} | {acc_str:<6} | {f1_str:<6} | {mcc_str:<6} | {auc_str:<6} | {latency_str:<10} | {far_str:<6} | {time_str:<25}")
```
with:
```python
        if row.get('Sensor') == 'best_ensemble' and index == df[df['Mode'] == row['Mode']].index[
            df[df['Mode'] == row['Mode']]['Sensor'].eq('best_ensemble').idxmax()
            if df[df['Mode'] == row['Mode']]['Sensor'].eq('best_ensemble').any()
            else 0
        ]:
            print(f"  {'--- BEST ENSEMBLE ---'}")
        print(f"{row['Model']:<30} | {acc_str:<6} | {f1_str:<6} | {mcc_str:<6} | {auc_str:<6} | {latency_str:<10} | {far_str:<6} | {time_str:<25}")
```

> Note: The separator logic above is complex. Use a simpler approach instead — track when sensor changes to 'best_ensemble' while iterating:

Actually replace the entire console output loop with this cleaner version:

```python
    current_mode = ""
    printed_ensemble_sep = False
    for index, row in df.iterrows():
        if row['Mode'] != current_mode:
            current_mode = row['Mode']
            printed_ensemble_sep = False
            print(f"\n--- {current_mode.upper()} MODE ---")
            print(f"{'Model':<30} | {'Sensor':<15} | {'Acc':<6} | {'F1':<6} | {'MCC':<6} | {'AUC':<6} | {'Latency':<10} | {'FAR':<6} | {'Timestamp':<25}")
            print("-" * 120)

        if row.get('Sensor') == 'best_ensemble' and not printed_ensemble_sep:
            print(f"  {'--- BEST ENSEMBLE ---'}")
            printed_ensemble_sep = True

        far_str = f"{row.get('FAR', float('nan')):.2%}" if pd.notna(row.get('FAR')) else "N/A"
        latency_str = f"{row.get('Latency_ms', float('nan')):.2f} ms" if pd.notna(row.get('Latency_ms')) else "N/A"
        time_str = str(row.get('Timestamp', 'Unknown')) if pd.notna(row.get('Timestamp')) else "Unknown"

        acc = row.get('Accuracy', float('nan'))
        f1 = row.get('F1-Score', float('nan'))
        mcc = row.get('MCC', float('nan'))
        auc = row.get('ROC-AUC', float('nan'))
        sensor = str(row.get('Sensor', 'unknown'))

        acc_str = f"{acc:.4f}" if pd.notna(acc) else "N/A"
        f1_str = f"{f1:.4f}" if pd.notna(f1) else "N/A"
        mcc_str = f"{mcc:.4f}" if pd.notna(mcc) else "N/A"
        auc_str = f"{auc:.4f}" if pd.notna(auc) else "N/A"

        print(f"{row['Model']:<30} | {sensor:<15} | {acc_str:<6} | {f1_str:<6} | {mcc_str:<6} | {auc_str:<6} | {latency_str:<10} | {far_str:<6} | {time_str:<25}")
```

- [ ] **Step 4: Sort ensemble rows to the bottom of each mode group**

Replace the sort line:
```python
    df = df.sort_values(
        by=["Mode", "F1-Score", "MCC"], ascending=[True, False, False]
    )
```
with:
```python
    df["_is_ensemble"] = (df["Sensor"] == "best_ensemble").astype(int)
    df = df.sort_values(
        by=["Mode", "_is_ensemble", "F1-Score", "MCC"],
        ascending=[True, True, False, False],
    )
    df = df.drop(columns=["_is_ensemble"])
```

- [ ] **Step 5: Verify aggregate_results.py imports cleanly**

```bash
poetry run python -c "import aggregate_results; print('aggregate_results.py imports OK')"
```
Expected: `aggregate_results.py imports OK`

- [ ] **Step 6: Commit**

```bash
git add aggregate_results.py
git commit -m "feat: add best_ensemble path scanning and leaderboard separator to aggregate_results.py"
```

---

## Task 7: End-to-End Verification

- [ ] **Step 1: Smoke test — model registry**

```bash
cd enforced-train
poetry run python -c "from models import MODEL_REGISTRY; assert 'TCN' not in MODEL_REGISTRY; assert 'IterativeMiniRocket' not in MODEL_REGISTRY; assert len(MODEL_REGISTRY) == 6; print('PASS:', sorted(MODEL_REGISTRY))"
```
Expected:
```
PASS: ['BiGRU', 'ClassificationCNN', 'ClassificationLSTM', 'DetectionCNN', 'InceptionTime', 'WaveformClassificationCNN']
```

- [ ] **Step 2: Smoke test — run_pipeline.sh syntax**

```bash
bash -n enforced-train/run_pipeline.sh && echo "PASS: no syntax errors"
```
Expected: `PASS: no syntax errors`

- [ ] **Step 3: Smoke test — eval.py and aggregate_results.py parse**

```bash
poetry run python -c "import eval; import aggregate_results; print('PASS: both import cleanly')"
```
Expected: `PASS: both import cleanly`

- [ ] **Step 4: Verify the `best_ensemble` scan path picks up reports correctly**

After at least one complete pipeline run that produces `saved_models/detection/best_ensemble/evaluation_report.txt`, run:
```bash
poetry run python aggregate_results.py
```
Confirm the console output includes a `--- BEST ENSEMBLE ---` separator line within the detection mode section, and that `master_evaluation_results.csv` contains rows with `Sensor == best_ensemble`.

- [ ] **Step 5: Final commit**

```bash
git add -p   # review any stray changes
git commit -m "chore: verify pipeline restructure complete — TCN/MiniRocket removed, parallel training, best-ensemble eval"
```
