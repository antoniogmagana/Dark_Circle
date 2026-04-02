import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import warnings

# Suppress the sklearn warning for undefined metrics (1 class present).
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# NOTICE: global 'config' is NO LONGER IMPORTED

from dataset import VehicleDataset
from models import build_model
from preprocess import preprocess_for_training


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
            f"  [!] Missing hyperparameters.json in {run_dir_path}. "
            f"Cannot reconstruct config."
        )
        return

    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    # JSON converts integer dictionary keys into strings.
    # We must revert CLASS_MAP keys back to integers so the confusion
    # matrix logic works.
    if "CLASS_MAP" in config_dict:
        config_dict["CLASS_MAP"] = {
            int(k): v for k, v in config_dict["CLASS_MAP"].items()
        }

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
    run_config.USE_MEL = meta.get(
        "use_mel", getattr(run_config, "USE_MEL", True)
    )

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
        config=run_config
    ).to(device)

    model_path = run_dir_path / "best_model.pth"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # Strip torch.compile prefix if the model was saved after compilation
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {
            k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()
        }
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

    precision = precision_score(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    unique_classes = len(np.unique(all_labels))
    if unique_classes > 1:
        try:
            if run_config.NUM_CLASSES == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr"
                )
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')

    target_labels = list(range(run_config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)

    far = None
    if run_config.TRAINING_MODE == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    # 8. Save Artifacts
    with open(report_path, "w") as f:
        f.write(f"Run Directory: {run_dir_path.name}\n")
        f.write(
            f"Mode: {run_config.TRAINING_MODE} "
            f"| Model: {run_config.MODEL_NAME}\n"
        )
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
                    f.write(
                        f"  {instance_name} ({k}): "
                        f"{per_class_acc[k]:.4f}\n"
                    )

    # ---------------------------------------------------------
    # Generate human-readable labels for the axes
    # ---------------------------------------------------------
    axis_labels = []
    if run_config.TRAINING_MODE == "detection":
        axis_labels = ["background", "target"]
    elif run_config.TRAINING_MODE == "category":
        axis_labels = [
            run_config.CLASS_MAP.get(i, str(i))
            for i in range(run_config.NUM_CLASSES)
        ]
    elif run_config.TRAINING_MODE == "instance":
        inv_map = {
            v: k
            for k, v in getattr(run_config, "INSTANCE_TO_CLASS", {}).items()
        }
        axis_labels = [
            inv_map.get(i, str(i)) for i in range(run_config.NUM_CLASSES)
        ]
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
        fmt='d',
        cmap='Blues',
        annot_kws={"size": annot_size, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
        xticklabels=axis_labels,
        yticklabels=axis_labels
    )

    plt.title(
        f"Confusion Matrix: {run_config.MODEL_NAME} "
        f"({run_config.TRAINING_MODE})",
        fontsize=26,
        pad=20,
    )
    plt.ylabel("True Label", fontsize=22, labelpad=14)
    plt.xlabel("Predicted Label", fontsize=22, labelpad=14)

    # Rotate the x-axis labels if there are a lot of them so they don't overlap
    if run_config.NUM_CLASSES > 5:
        plt.xticks(rotation=45, ha='right', fontsize=20)
    else:
        plt.xticks(rotation=0, fontsize=20)

    plt.yticks(rotation=0, fontsize=20)

    # Scale up the colorbar tick labels too
    plt.gcf().axes[-1].tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(
        run_dir_path / "conf_matrix.png", dpi=300, bbox_inches='tight'
    )
    plt.close()


def evaluate_fused(mode_dir):
    """
    Fuse per-sensor models for one training mode via weighted average softmax.
    Expects directory structure: saved_models/{mode}/{sensor}/{model}/{run_id}/
    Groups runs by (model_name, run_id) across sensors, weights by val_f1.
    Saves evaluation_report.txt to saved_models/{mode}/fused/{model}_{run_id}/.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Collect all per-sensor run dirs under this mode
    sensor_run_dirs = [p.parent for p in mode_dir.rglob("best_model.pth")]
    if not sensor_run_dirs:
        return

    # Group by (model_name, run_id) — {mode}/{sensor}/{model}/{run_id}
    groups = {}
    for rd in sensor_run_dirs:
        run_id = rd.name
        model_name = rd.parent.name
        sensor = rd.parent.parent.name
        key = (model_name, run_id)
        groups.setdefault(key, []).append((sensor, rd))

    for (model_name, run_id), sensor_dirs in groups.items():
        if len(sensor_dirs) < 2:
            continue  # need at least 2 sensors to fuse

        fused_dir = mode_dir / "fused" / f"{model_name}_{run_id}"
        report_path = fused_dir / "evaluation_report.txt"
        if report_path.exists():
            print(f"Skipping fused (exists): {fused_dir}")
            continue

        print(
            f"\nFused eval: mode={mode_dir.name} "
            f"model={model_name} run={run_id}"
        )

        # Load config and build dataset from the first sensor's run
        _, first_rd = sensor_dirs[0]
        json_path = first_rd / "hyperparameters.json"
        if not json_path.exists():
            print(f"  [!] Missing hyperparameters.json in {first_rd}")
            continue
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        if "CLASS_MAP" in config_dict:
            config_dict["CLASS_MAP"] = {
                int(k): v for k, v in config_dict["CLASS_MAP"].items()
            }
        run_config = SimpleNamespace(**config_dict)
        run_config.DEVICE = device_str

        # Build test dataset once (sensor-agnostic; labels/splits shared)
        test_ds = VehicleDataset(split="test", config=run_config)
        if len(test_ds) == 0:
            print(f"  [!] No test samples for {run_config.TRAINING_MODE}.")
            continue
        test_loader = DataLoader(
            test_ds,
            batch_size=run_config.BATCH_SIZE,
            shuffle=False,
            num_workers=run_config.NUM_WORKERS,
        )

        # Load each sensor model and its val_f1 weight
        models_weights = []
        for sensor, rd in sensor_dirs:
            meta_path = rd / "meta.pt"
            model_path = rd / "best_model.pth"
            if not meta_path.exists() or not model_path.exists():
                print(
                    f"  [!] Missing artifacts for sensor={sensor}, skipping."
                )
                continue
            meta = torch.load(
                meta_path, map_location=device, weights_only=False
            )
            val_f1 = meta.get("val_f1", 0.0)

            # Reconstruct per-sensor config for IN_CHANNELS
            s_json = rd / "hyperparameters.json"
            with open(s_json, "r") as f:
                s_cfg = SimpleNamespace(**json.load(f))
            if hasattr(s_cfg, "CLASS_MAP"):
                s_cfg.CLASS_MAP = {
                    int(k): v for k, v in s_cfg.CLASS_MAP.items()
                }
            s_cfg.USE_MEL = meta.get(
                "use_mel", getattr(s_cfg, "USE_MEL", False)
            )

            m = build_model(
                input_channels=s_cfg.IN_CHANNELS,
                num_classes=s_cfg.NUM_CLASSES,
                config=s_cfg,
            ).to(device)
            state_dict = torch.load(
                model_path, map_location=device, weights_only=True
            )
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {
                    k.removeprefix("_orig_mod."): v
                    for k, v in state_dict.items()
                }
            m.load_state_dict(state_dict)
            m.eval()
            models_weights.append((sensor, m, s_cfg, val_f1))

        if len(models_weights) < 2:
            print(
                "  [!] Fewer than 2 sensor models loaded; "
                "skipping fused eval."
            )
            continue

        # Normalize weights by val_f1
        total_f1 = sum(w for _, _, _, w in models_weights) or 1.0
        weights = [w / total_f1 for _, _, _, w in models_weights]

        # Inference: each model gets its own preprocessed input
        all_fused_preds = []
        all_labels = []
        all_fused_probs = []

        start_time = time.perf_counter()
        with torch.inference_mode():
            for batch in test_loader:
                x, y = batch[0], batch[1]
                y_np = y.numpy()
                fused_probs = None
                for (sensor, m, s_cfg, _), w in zip(models_weights, weights):
                    xi = x.to(device)
                    xi = preprocess_for_training(xi, config=s_cfg)
                    logits = m(xi)
                    probs = F.softmax(logits, dim=1).cpu()
                    fused_probs = (
                        probs * w if fused_probs is None
                        else fused_probs + probs * w
                    )
                preds = fused_probs.argmax(dim=1).numpy()
                all_fused_preds.extend(preds)
                all_labels.extend(y_np)
                all_fused_probs.extend(fused_probs.numpy())
        end_time = time.perf_counter()

        all_labels = np.array(all_labels)
        all_fused_preds = np.array(all_fused_preds)
        all_fused_probs = np.array(all_fused_probs)
        latency_ms = ((end_time - start_time) / len(test_ds)) * 1000

        acc = accuracy_score(all_labels, all_fused_preds)
        mcc = matthews_corrcoef(all_labels, all_fused_preds)
        precision = precision_score(
            all_labels, all_fused_preds, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_fused_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(
            all_labels, all_fused_preds, average="weighted", zero_division=0
        )

        unique_classes = len(np.unique(all_labels))
        if unique_classes > 1:
            try:
                if run_config.NUM_CLASSES == 2:
                    auc = roc_auc_score(all_labels, all_fused_probs[:, 1])
                else:
                    auc = roc_auc_score(
                        all_labels, all_fused_probs, multi_class="ovr"
                    )
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")

        target_labels = list(range(run_config.NUM_CLASSES))
        cm = confusion_matrix(
            all_labels, all_fused_preds, labels=target_labels
        )
        far = None
        if run_config.TRAINING_MODE == "detection" and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
            per_class_acc[np.isnan(per_class_acc)] = 0.0

        fused_dir.mkdir(parents=True, exist_ok=True)
        sensor_names = "+".join(s for s, _, _, _ in models_weights)
        with open(report_path, "w") as f:
            f.write(f"Run Directory: {fused_dir.name}\n")
            f.write(
                f"Mode: {run_config.TRAINING_MODE} | Model: {model_name} "
                f"[fused: {sensor_names}]\n"
            )
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
                inv_map = {
                    v: k for k, v in run_config.INSTANCE_TO_CLASS.items()
                }
                for k in range(run_config.NUM_CLASSES):
                    if k < len(per_class_acc):
                        name = inv_map.get(k, f"Class_{k}")
                        f.write(f"  {name} ({k}): {per_class_acc[k]:.4f}\n")
            f.write(
                f"\nFusion weights: "
                + ", ".join(
                    f"{s}={w:.4f}"
                    for (s, _, _, _), w in zip(models_weights, weights)
                )
                + "\n"
            )
        print(f"  Fused report saved to {report_path}")


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


if __name__ == "__main__":
    main()
