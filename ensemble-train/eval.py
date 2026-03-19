import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from functools import partial
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
    f1_score,
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from dataset import VehicleDataset, db_worker_init
from models import build_model
from preprocess import preprocess


# =====================================================================
# Helpers
# =====================================================================

def build_axis_labels(run_config):
    """Human-readable labels for confusion matrix axes."""
    if run_config.TRAINING_MODE == "detection":
        return ["background", "target"]
    elif run_config.TRAINING_MODE == "category":
        return [run_config.CLASS_MAP.get(i, str(i)) for i in range(run_config.NUM_CLASSES)]
    elif run_config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in getattr(run_config, "INSTANCE_TO_CLASS", {}).items()}
        return [inv_map.get(i, str(i)) for i in range(run_config.NUM_CLASSES)]
    return [str(i) for i in range(run_config.NUM_CLASSES)]


def compute_auc(labels, probs, num_classes):
    """Safely compute ROC-AUC, returning NaN when undefined."""
    if len(np.unique(labels)) <= 1:
        return float("nan")
    try:
        if num_classes == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        return float("nan")


# =====================================================================
# Main Evaluation
# =====================================================================

def evaluate_directory(run_dir_path):
    report_path = run_dir_path / "evaluation_report.txt"
    if report_path.exists():
        print(f"Skipping: {run_dir_path} (already evaluated)")
        return

    print(f"\nEvaluating: {run_dir_path}")

    # ------------------------------------------------------------------
    # 1. Reconstruct config from JSON snapshot
    # ------------------------------------------------------------------
    json_path = run_dir_path / "hyperparameters.json"
    if not json_path.exists():
        print(f"  [!] Missing hyperparameters.json in {run_dir_path}")
        return

    with open(json_path) as f:
        config_dict = json.load(f)

    if "CLASS_MAP" in config_dict:
        config_dict["CLASS_MAP"] = {int(k): v for k, v in config_dict["CLASS_MAP"].items()}

    run_config = SimpleNamespace(**config_dict)
    device = torch.device(getattr(run_config, "DEVICE", "cpu"))

    # ------------------------------------------------------------------
    # 2. Load metadata (USE_MEL flag)
    # ------------------------------------------------------------------
    meta_path = run_dir_path / "meta.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, map_location=device, weights_only=False)
        run_config.USE_MEL = meta.get("use_mel", getattr(run_config, "USE_MEL", True))

    # ------------------------------------------------------------------
    # 3. Dataset & model
    # ------------------------------------------------------------------
    test_ds = VehicleDataset(split="test", config=run_config)
    if len(test_ds) == 0:
        print(f"  [!] No test samples for mode={run_config.TRAINING_MODE}")
        return

    custom_worker_init = partial(db_worker_init, config=run_config)
    test_loader = DataLoader(
        test_ds,
        batch_size=run_config.BATCH_SIZE,
        shuffle=False,
        num_workers=run_config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
    )

    model = build_model(
        input_channels=run_config.IN_CHANNELS,
        num_classes=run_config.NUM_CLASSES,
        config=run_config,
    ).to(device)

    model_path = run_dir_path / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    all_preds, all_labels, all_probs = [], [], []

    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _dataset_names = batch
            else:
                x, y = batch

            x = x.to(device)
            x = preprocess(x, config=run_config)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    elapsed = time.perf_counter() - start_time

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    latency_ms = (elapsed / len(test_ds)) * 1000
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    auc = compute_auc(all_labels, all_probs, run_config.NUM_CLASSES)

    target_labels = list(range(run_config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)

    far = None
    if run_config.TRAINING_MODE == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    # ------------------------------------------------------------------
    # 6. Save predictions (for ensemble weight tuning)
    # ------------------------------------------------------------------
    np.savez_compressed(
        run_dir_path / "predictions.npz",
        labels=all_labels,
        probs=all_probs,
    )

    # ------------------------------------------------------------------
    # 7. Save text report
    # ------------------------------------------------------------------
    with open(report_path, "w") as f:
        f.write(f"Run Directory: {run_dir_path.name}\n")
        f.write(f"Mode: {run_config.TRAINING_MODE} | Model: {run_config.MODEL_NAME}")
        if hasattr(run_config, "TRAIN_SENSOR"):
            f.write(f" | Sensor: {run_config.TRAIN_SENSOR}")
        f.write("\n")
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
        axis_labels = build_axis_labels(run_config)
        for i, name in enumerate(axis_labels):
            if i < len(per_class_acc):
                f.write(f"  {name} ({i}): {per_class_acc[i]:.4f}\n")

    # ------------------------------------------------------------------
    # 8. Confusion matrix plot
    # ------------------------------------------------------------------
    axis_labels = build_axis_labels(run_config)
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
        f"Confusion Matrix: {run_config.MODEL_NAME} ({run_config.TRAINING_MODE})",
        fontsize=26, pad=20,
    )
    plt.ylabel("True Label", fontsize=22, labelpad=14)
    plt.xlabel("Predicted Label", fontsize=22, labelpad=14)

    rotation = 45 if run_config.NUM_CLASSES > 5 else 0
    ha = "right" if rotation else "center"
    plt.xticks(rotation=rotation, ha=ha, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.gcf().axes[-1].tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(run_dir_path / "conf_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = Path("saved_models")
    if not base_dir.exists():
        print("No saved_models directory found.")
        return

    run_dirs = sorted({p.parent for p in base_dir.rglob("best_model.pth")})
    for run_dir in run_dirs:
        evaluate_directory(run_dir)


if __name__ == "__main__":
    main()
