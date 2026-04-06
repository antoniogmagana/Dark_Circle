"""
CRL evaluation script.

Two evaluation passes:

  Eval A — Val set (focal + m3nvc):
    - Detection:     2-class confusion matrix + metrics
    - Classification: 4-class confusion matrix + metrics

  Eval B — IOBT test set:
    - Detection only (present=False windows serve as negatives)
    - Classification is skipped (all iobt vehicles are the same category)

Outputs (written to --save-dir):
  detection_report_val.txt / detection_report_iobt.txt
  classification_report_val.txt
  conf_matrix_detection_val.png / conf_matrix_detection_iobt.png
  conf_matrix_classification_val.png
  metrics_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader

from causal_dataset import MultiModalCausalDataset, collate_multimodal
from causal_vae import MultiModalCausalVAE
from train_crl import DetectionHead, ClassificationHead
from crl_config import (
    CLASS_MAP,
    CLASS_NAMES,
    Z_VEH_DIM,
    Z_ENV_DIM,
    MODALITY_FEATURE_DIM,
    BATCH_SIZE,
    NUM_WORKERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(batch_t: dict, device: torch.device) -> dict:
    return {m: v.to(device) if v is not None else None for m, v in batch_t.items()}


def _plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {path}")


def _write_report(lines: list, path: Path):
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved report → {path}")


# ---------------------------------------------------------------------------
# Detection evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_detection_eval(
    model: MultiModalCausalVAE,
    det_head: DetectionHead,
    loader: DataLoader,
    device: torch.device,
    eval_set_name: str,
    save_dir: Path,
) -> dict:
    """
    Returns dict of detection metrics and saves confusion matrix + report.
    """
    model.eval()
    det_head.eval()

    all_preds, all_true, all_probs = [], [], []

    for batch in loader:
        batch_t, _, avail, _, _cat_labels, det_labels = batch
        batch_t = _to_device(batch_t, device)
        avail = avail.to(device)

        z_veh = model.encode_veh(batch_t, avail)
        logits = det_head(z_veh)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(vehicle)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_true.extend(det_labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    labels_list = ["background", "vehicle"]
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(all_true, all_preds),
        "f1": f1_score(all_true, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(
            all_true, all_preds, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            all_true, all_preds, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(all_true, all_preds),
        "far": far,
    }
    try:
        metrics["auc"] = roc_auc_score(all_true, all_probs)
    except ValueError:
        metrics["auc"] = float("nan")

    # Confusion matrix plot
    _plot_confusion_matrix(
        cm,
        labels_list,
        title=f"Detection — {eval_set_name}",
        path=save_dir / f"conf_matrix_detection_{eval_set_name}.png",
    )

    # Text report
    lines = [
        f"=== Detection Evaluation: {eval_set_name} ===",
        f"Samples   : {len(all_true)}",
        f"Accuracy  : {metrics['accuracy']:.4f}",
        f"F1 (wtd)  : {metrics['f1']:.4f}",
        f"Precision : {metrics['precision']:.4f}",
        f"Recall    : {metrics['recall']:.4f}",
        f"AUC-ROC   : {metrics['auc']:.4f}",
        f"FAR       : {metrics['far']:.4f}",
        f"MCC       : {metrics['mcc']:.4f}",
        "",
        "Confusion Matrix (rows=true, cols=pred):",
        f"  labels: {labels_list}",
        str(cm),
    ]
    _write_report(lines, save_dir / f"detection_report_{eval_set_name}.txt")

    return metrics


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_classification_eval(
    model: MultiModalCausalVAE,
    cls_head: ClassificationHead,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
) -> dict:
    """
    4-class classification evaluation on the val set.
    Background and multi-vehicle windows are excluded.

    NOTE: Do not call this on the iobt test set — all iobt vehicles
    are the same category, making per-class metrics meaningless.
    """
    model.eval()
    cls_head.eval()

    all_preds, all_true = [], []

    for batch in loader:
        batch_t, _, avail, _, cat_labels, _ = batch
        cls_mask = cat_labels >= 0
        if not cls_mask.any():
            continue

        batch_t = _to_device(batch_t, device)
        avail = avail.to(device)

        z_veh = model.encode_veh(batch_t, avail)
        z_masked = z_veh[cls_mask.to(device)]
        logits = cls_head(z_masked)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_true.extend(cat_labels[cls_mask].tolist())

    if not all_true:
        print("  WARNING: no labelled classification samples found in loader.")
        return {}

    target_labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(all_true, all_preds, labels=target_labels)

    per_class_acc = np.zeros(len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        row_sum = cm[i].sum()
        per_class_acc[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(all_true, all_preds),
        "macro_f1": f1_score(all_true, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            all_true, all_preds, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(all_true, all_preds),
        "per_class_acc": per_class_acc.tolist(),
        "per_class_f1": f1_score(
            all_true,
            all_preds,
            average=None,
            labels=target_labels,
            zero_division=0,
        ).tolist(),
    }

    # Confusion matrix plot
    _plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        title="Classification — Val Set",
        path=save_dir / "conf_matrix_classification_val.png",
    )

    # Text report
    lines = [
        "=== Classification Evaluation: Val Set ===",
        f"Samples      : {len(all_true)}",
        f"Accuracy     : {metrics['accuracy']:.4f}",
        f"Macro F1     : {metrics['macro_f1']:.4f}",
        f"Weighted F1  : {metrics['weighted_f1']:.4f}",
        f"MCC          : {metrics['mcc']:.4f}",
        "",
        "Per-class accuracy:",
    ]
    for name, acc, f1 in zip(
        CLASS_NAMES, metrics["per_class_acc"], metrics["per_class_f1"]
    ):
        lines.append(f"  {name:12s}: acc={acc:.4f}  f1={f1:.4f}")
    lines += [
        "",
        "sklearn classification_report:",
        classification_report(
            all_true,
            all_preds,
            labels=target_labels,
            target_names=CLASS_NAMES,
            zero_division=0,
        ),
        "Confusion Matrix (rows=true, cols=pred):",
        f"  labels: {CLASS_NAMES}",
        str(cm),
    ]
    _write_report(lines, save_dir / "classification_report_val.txt")

    return metrics


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------


def write_summary(results: dict, save_dir: Path):
    path = save_dir / "metrics_summary.csv"
    fieldnames = [
        "eval_set",
        "task",
        "accuracy",
        "f1_weighted",
        "macro_f1",
        "precision",
        "recall",
        "auc",
        "far",
        "mcc",
    ]
    rows = []

    for eval_set, tasks in results.items():
        for task, m in tasks.items():
            rows.append(
                {
                    "eval_set": eval_set,
                    "task": task,
                    "accuracy": f"{m.get('accuracy', float('nan')):.4f}",
                    "f1_weighted": f"{m.get('f1', m.get('weighted_f1', float('nan'))):.4f}",
                    "macro_f1": f"{m.get('macro_f1', float('nan')):.4f}",
                    "precision": f"{m.get('precision', float('nan')):.4f}",
                    "recall": f"{m.get('recall', float('nan')):.4f}",
                    "auc": f"{m.get('auc', float('nan')):.4f}",
                    "far": f"{m.get('far', float('nan')):.4f}",
                    "mcc": f"{m.get('mcc', float('nan')):.4f}",
                }
            )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved metrics summary → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CRL Evaluation")
    p.add_argument("--val-dir", required=True, help="Path to val parquet files")
    p.add_argument("--iobt-dir", required=True, help="Path to iobt test parquet files")
    p.add_argument(
        "--save-dir", default="./saved_crl", help="Directory with saved model weights"
    )
    p.add_argument(
        "--eval-dir", default="./eval_crl", help="Output directory for eval results"
    )
    p.add_argument("--modalities", nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--z-veh-dim", type=int, default=Z_VEH_DIM)
    p.add_argument("--z-env-dim", type=int, default=Z_ENV_DIM)
    return p.parse_args()


def load_model(
    save_dir: Path,
    num_domains: int,
    z_veh_dim: int,
    z_env_dim: int,
    device: torch.device,
):
    model = MultiModalCausalVAE(
        num_sensor_domains=num_domains,
        modality_feat_dim=MODALITY_FEATURE_DIM,
        z_veh_dim=z_veh_dim,
        z_env_dim=z_env_dim,
    ).to(device)

    det_head = DetectionHead(z_veh_dim=z_veh_dim).to(device)
    cls_head = ClassificationHead(z_veh_dim=z_veh_dim, num_classes=len(CLASS_MAP)).to(
        device
    )

    def _load(name):
        p = save_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return torch.load(p, map_location=device)

    model.load_state_dict(_load("crl_best.pth"))
    det_head.load_state_dict(_load("det_head_best.pth"))
    cls_head.load_state_dict(_load("cls_head_best.pth"))
    return model, det_head, cls_head


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Build datasets (filter_present=False so all windows are evaluated)
    val_ds = MultiModalCausalDataset(
        parquet_dir=args.val_dir,
        filter_present=False,
        include_modalities=args.modalities,
        domain_map={"__UNKNOWN__": 0},  # Force unseen mapping for zero-shot eval
    )
    iobt_ds = MultiModalCausalDataset(
        parquet_dir=args.iobt_dir,
        filter_present=False,
        include_modalities=args.modalities,
        domain_map={"__UNKNOWN__": 0},
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    iobt_loader = DataLoader(iobt_ds, shuffle=False, **loader_kwargs)

    # Load trained model's domain count
    meta_path = save_dir / "crl_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            num_domains = json.load(f)["num_domains"]
    else:
        num_domains = val_ds.num_sensor_domains

    model, det_head, cls_head = load_model(
        save_dir, num_domains, args.z_veh_dim, args.z_env_dim, device
    )

    results = {}

    # --- Eval A: Val set ---
    print("\n--- Eval A: Val Set ---")
    val_det_metrics = run_detection_eval(
        model,
        det_head,
        val_loader,
        device,
        eval_set_name="val",
        save_dir=eval_dir,
    )
    val_cls_metrics = run_classification_eval(
        model,
        cls_head,
        val_loader,
        device,
        save_dir=eval_dir,
    )
    results["val"] = {"detection": val_det_metrics, "classification": val_cls_metrics}

    # --- Eval B: IOBT test set ---
    print("\n--- Eval B: IOBT Test Set ---")
    iobt_det_metrics = run_detection_eval(
        model,
        det_head,
        iobt_loader,
        device,
        eval_set_name="iobt",
        save_dir=eval_dir,
    )
    print(
        "  NOTE: Classification eval skipped for iobt "
        "(all vehicles are same category — metrics would be uninformative)."
    )
    results["iobt"] = {"detection": iobt_det_metrics}

    # --- Summary ---
    write_summary(results, eval_dir)

    print("\n=== Summary ===")
    for eval_set, tasks in results.items():
        for task, m in tasks.items():
            f1 = m.get("f1", m.get("weighted_f1", float("nan")))
            print(
                f"  {eval_set:6s} {task:14s}  acc={m.get('accuracy', float('nan')):.4f}"
                f"  F1={f1:.4f}  MCC={m.get('mcc', float('nan')):.4f}"
            )


if __name__ == "__main__":
    main()
