import os
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
            "Cannot reconstruct config."
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

    # Convert dictionary to a dot-accessible object that mimics the
    # 'config' module
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
        print(
            f"  [!] No test samples found for {run_config.TRAINING_MODE}."
        )
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
    state_dict = torch.load(
        model_path, map_location=device, weights_only=True
    )
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

    # 8. Save evaluation report
    with open(report_path, "w") as f:
        f.write(f"Run Directory: {run_dir_path.name}\n")
        f.write(
            f"Mode: {run_config.TRAINING_MODE} | "
            f"Model: {run_config.MODEL_NAME}\n"
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
                    instance_name = inv_map.get(k, f"Class_{k}")
                    f.write(
                        f"  {instance_name} ({k}): {per_class_acc[k]:.4f}\n"
                    )

    # 9. Save raw predictions for ensemble aggregation
    np.savez(
        run_dir_path / "predictions.npz",
        labels=all_labels,
        preds=all_preds,
        probs=all_probs,
    )

    # 10. Build axis labels for confusion matrix
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

    # 11. Plot confusion matrix
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

    if run_config.NUM_CLASSES > 5:
        plt.xticks(rotation=45, ha='right', fontsize=20)
    else:
        plt.xticks(rotation=0, fontsize=20)

    plt.yticks(rotation=0, fontsize=20)
    plt.gcf().axes[-1].tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(
        run_dir_path / "conf_matrix.png", dpi=300, bbox_inches='tight'
    )
    plt.close()


def main():
    base_dir = Path("saved_models")
    if not base_dir.exists():
        print("No saved_models directory found.")
        return

    run_dirs = [p.parent for p in base_dir.rglob("best_model.pth")]

    for run_dir in run_dirs:
        evaluate_directory(run_dir)


if __name__ == "__main__":
    main()
