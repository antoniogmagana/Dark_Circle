import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

# Suppress the sklearn warning for undefined metrics (1 class present).
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# NOTICE: global 'config' is NO LONGER IMPORTED

from dataset import VehicleDataset
from preprocess import preprocess_for_training

from models import build_model


def evaluate_directory(run_dir_path):
    # 1. Check for existing artifacts
    report_path = run_dir_path / "evaluation_report.txt"
    if report_path.exists():
        print(f"Skipping: {run_dir_path} (Artifacts already exist)")
        return

    print(f"\nEvaluating: {run_dir_path}")

    # 2. Reconstruct the Run's Configuration from its JSON Snapshot
    json_path = run_dir_path / "hyperparameters.json"
    if not json_path.exists():
        print(
            f"  [!] Missing hyperparameters.json in {run_dir_path}. " f"Cannot reconstruct config."
        )
        return

    with open(json_path) as f:
        config_dict = json.load(f)

    # JSON converts integer dictionary keys into strings.
    # We must revert CLASS_MAP keys back to integers so the confusion
    # matrix logic works.
    if "CLASS_MAP" in config_dict:
        config_dict["CLASS_MAP"] = {int(k): v for k, v in config_dict["CLASS_MAP"].items()}

    # Convert dictionary to a dot-accessible object that mimics 'config'
    run_config = SimpleNamespace(**config_dict)

    device_str = getattr(run_config, "DEVICE", "cpu")
    device = torch.device(device_str)

    # 3. Load Metadata Tensors
    meta_path = run_dir_path / "meta.pt"
    if not meta_path.exists():
        print(f"  [!] Missing meta.pt in {run_dir_path}")
        return

    meta = torch.load(meta_path, map_location=device, weights_only=False)

    # Force USE_MEL to whatever was saved in the metadata
    run_config.USE_MEL = meta.get("use_mel", getattr(run_config, "USE_MEL", True))

    # 4. Build Dataset & DataLoader using the injected run_config
    test_ds = VehicleDataset(split="test", config=run_config)

    if len(test_ds) == 0:
        print(f"  [!] No test samples found for {run_config.TRAINING_MODE}.")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=run_config.BATCH_SIZE,
        shuffle=False,
        num_workers=run_config.NUM_WORKERS,
    )

    # 5. Load Model
    model = build_model(
        input_channels=run_config.IN_CHANNELS,
        num_classes=run_config.NUM_CLASSES,
        config=run_config,
    ).to(device)

    model_path = run_dir_path / "best_model.pth"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # Strip torch.compile prefix if the model was saved after compilation
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 6. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = []

    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in test_loader:
            # Safely unpack based on whether the dataset returned the name
            if len(batch) == 3:
                x, y, dataset_names = batch
            else:
                x, y = batch

            x = x.to(device)

            x = preprocess_for_training(x, config=run_config)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    end_time = time.perf_counter()

    # 7. Calculate Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    latency_ms = ((end_time - start_time) / len(test_ds)) * 1000

    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    unique_classes = len(np.unique(all_labels))
    if unique_classes > 1:
        try:
            if run_config.NUM_CLASSES == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    target_labels = list(range(run_config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)

    far = None
    if run_config.TRAINING_MODE == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    # 8. Save Artifacts
    with open(report_path, "w") as f:
        f.write(f"Run Directory: {run_dir_path.name}\n")
        f.write(f"Mode: {run_config.TRAINING_MODE} " f"| Model: {run_config.MODEL_NAME}\n")
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
        if run_config.TRAINING_MODE == "detection":
            f.write(f"  Background (0): {per_class_acc[0]:.4f}\n")
            f.write(f"  Target (1): {per_class_acc[1]:.4f}\n")
        elif run_config.TRAINING_MODE == "category":
            for k, v in run_config.CLASS_MAP.items():
                if k < len(per_class_acc):
                    f.write(f"  {v} ({k}): {per_class_acc[k]:.4f}\n")
        elif run_config.TRAINING_MODE == "instance":
            # Reverse the dictionary to look up names by their integer ID
            inv_map = {v: k for k, v in run_config.INSTANCE_TO_CLASS.items()}
            for k in range(run_config.NUM_CLASSES):
                if k < len(per_class_acc):
                    instance_name = inv_map.get(k, f"Class_{k}")
                    f.write(f"  {instance_name} ({k}): " f"{per_class_acc[k]:.4f}\n")

    # ---------------------------------------------------------
    # Generate human-readable labels for the axes
    # ---------------------------------------------------------
    axis_labels = []
    if run_config.TRAINING_MODE == "detection":
        axis_labels = ["background", "target"]
    elif run_config.TRAINING_MODE == "category":
        axis_labels = [run_config.CLASS_MAP.get(i, str(i)) for i in range(run_config.NUM_CLASSES)]
    elif run_config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in getattr(run_config, "INSTANCE_TO_CLASS", {}).items()}
        axis_labels = [inv_map.get(i, str(i)) for i in range(run_config.NUM_CLASSES)]
    else:
        axis_labels = [str(i) for i in range(run_config.NUM_CLASSES)]

    # ---------------------------------------------------------
    # Plot Heatmap with Dynamic Scaling & String Labels
    # ---------------------------------------------------------
    fig_size = max(12, run_config.NUM_CLASSES * 1.2)
    annot_size = max(18, min(26, int(240 / run_config.NUM_CLASSES)))

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
        f"Confusion Matrix: {run_config.MODEL_NAME} " f"({run_config.TRAINING_MODE})",
        fontsize=26,
        pad=20,
    )
    plt.ylabel("True Label", fontsize=22, labelpad=14)
    plt.xlabel("Predicted Label", fontsize=22, labelpad=14)

    # Rotate the x-axis labels if there are a lot of them so they don't overlap
    if run_config.NUM_CLASSES > 5:
        plt.xticks(rotation=45, ha="right", fontsize=20)
    else:
        plt.xticks(rotation=0, fontsize=20)

    plt.yticks(rotation=0, fontsize=20)

    # Scale up the colorbar tick labels too
    plt.gcf().axes[-1].tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(run_dir_path / "conf_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


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
            val_f1 = meta.get("val_f1", -1.0)
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
        with open(json_path) as f:
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
        print("  [!] Could not load one or both models for best_ensemble. Skipping.")
        return

    # Build separate test DataLoaders — preprocessing differs between sensors
    seismic_test_ds = VehicleDataset(split="test", config=seismic_cfg)
    audio_test_ds = VehicleDataset(split="test", config=audio_cfg)

    if len(seismic_test_ds) == 0 or len(audio_test_ds) == 0:
        print("  [!] Empty test dataset for best_ensemble. Skipping.")
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

    # Use the seismic config as the reference (same label space for both)
    ref_cfg = seismic_cfg

    start_time = time.perf_counter()
    seismic_probs, seismic_labels = _run_inference(seismic_model, seismic_loader, seismic_cfg)
    audio_probs, audio_labels = _run_inference(audio_model, audio_loader, audio_cfg)
    end_time = time.perf_counter()

    # Sanity check: both loaders must yield the same labels in the same order
    if not torch.equal(seismic_labels, audio_labels):
        print(
            "  [!] Label mismatch between seismic and audio test sets. "
            "Cannot produce a valid ensemble. Skipping."
        )
        return

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
        f.write("Run Directory: best_ensemble\n")
        f.write(
            f"Mode: {ref_cfg.TRAINING_MODE} | Model: {combined_model_label} " f"[best_ensemble]\n"
        )
        f.write(
            f"Seismic: {seismic_model_name} (run: {seismic_run_dir.name}, val_f1={seismic_f1:.4f})\n"
        )
        f.write(f"Audio:   {audio_model_name} (run: {audio_run_dir.name}, val_f1={audio_f1:.4f})\n")
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

    # Confusion matrix plot (same style as evaluate_directory)
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


def main():
    base_dir = Path("saved_models")
    if not base_dir.exists():
        print("No saved_models directory found.")
        return

    # Per-sensor evaluation — skip runs inside best_ensemble dirs
    run_dirs = [
        p.parent for p in base_dir.rglob("best_model.pth") if "best_ensemble" not in p.parts
    ]
    for run_dir in run_dirs:
        evaluate_directory(run_dir)

    # Best-ensemble evaluation: one pass per training mode directory
    mode_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name not in ("cache",)]
    for mode_dir in mode_dirs:
        evaluate_best_ensemble(mode_dir)


if __name__ == "__main__":
    main()
