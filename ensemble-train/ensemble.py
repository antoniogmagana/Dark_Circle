"""
Ensemble evaluation — polls all trained models for a given mode and
combines their softmax outputs into a single pooled prediction.

Each model's prediction is weighted by its individual test F1-Score.
The final prediction is the argmax of the weighted sum of softmax
probability vectors (a valid probability distribution).

Usage:
    TRAINING_MODE=detection poetry run python ensemble.py detection
    TRAINING_MODE=category  poetry run python ensemble.py category
    TRAINING_MODE=instance  poetry run python ensemble.py instance
    poetry run python ensemble.py          # runs all three modes
"""

import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
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


# =====================================================================
# 1. Discovery — find the best evaluated run per model architecture
# =====================================================================

def _parse_f1_from_report(report_path):
    """Return the F1-Score float from an evaluation_report.txt, or None."""
    try:
        with open(report_path) as f:
            for line in f:
                if line.startswith("F1-Score:"):
                    return float(line.split(":")[1].strip())
    except Exception:
        pass
    return None


def discover_members(mode, base_dir="saved_models"):
    """
    Scan saved_models/{mode}/{model_name}/*/evaluation_report.txt.
    For each model architecture, keep only the run with the highest F1.

    Returns a list of dicts:
        {model_name, run_id, run_dir (Path), f1}
    """
    pattern = Path(base_dir) / mode / "*" / "*" / "evaluation_report.txt"
    candidates = {}

    for report_path in Path(base_dir).glob(f"{mode}/*/*/evaluation_report.txt"):
        parts = report_path.parts
        # structure: saved_models / mode / model_name / run_id / report
        model_name = parts[-3]
        run_id = parts[-2]
        run_dir = report_path.parent

        f1 = _parse_f1_from_report(report_path)
        if f1 is None:
            continue

        # Also require predictions.npz to exist (written by eval.py)
        if not (run_dir / "predictions.npz").exists():
            print(
                f"  [!] Skipping {run_dir}: predictions.npz missing. "
                "Re-run eval.py to generate it."
            )
            continue

        if model_name not in candidates or f1 > candidates[model_name]["f1"]:
            candidates[model_name] = {
                "model_name": model_name,
                "run_id": run_id,
                "run_dir": run_dir,
                "f1": f1,
            }

    return list(candidates.values())


# =====================================================================
# 2. Weight computation
# =====================================================================

def compute_weights(members):
    """
    Linear F1-normalised weights: w_i = f1_i / sum(f1_j).
    Falls back to uniform weights if all F1s are zero.
    Adds a 'weight' key to each member dict in-place.
    """
    total = sum(m["f1"] for m in members)
    for m in members:
        m["weight"] = (m["f1"] / total) if total > 0 else 1.0 / len(members)
    return members


# =====================================================================
# 3. Ensemble evaluation (offline — from predictions.npz)
# =====================================================================

def evaluate_ensemble(members, mode):
    """
    Load predictions.npz from each member, pool softmax probabilities
    by weighted sum, then compute the full metric suite.

    Returns a result dict with keys:
        labels, pooled_preds, pooled_probs,
        ensemble_metrics, individual_metrics
    """
    labels_ref = None
    pooled_probs = None

    individual_metrics = []

    for m in members:
        data = np.load(m["run_dir"] / "predictions.npz")
        labels = data["labels"]
        probs = data["probs"]   # [N, C]
        preds = data["preds"]

        # Verify all models evaluated on the same test set
        if labels_ref is None:
            labels_ref = labels
        else:
            if not np.array_equal(labels, labels_ref):
                raise ValueError(
                    f"Label mismatch between models in mode '{mode}'. "
                    "Ensure all models were trained with identical splits."
                )

        # Per-model metrics
        f1_ind = f1_score(labels, preds, average="weighted", zero_division=0)
        acc_ind = accuracy_score(labels, preds)
        mcc_ind = matthews_corrcoef(labels, preds)
        prec_ind = precision_score(
            labels, preds, average="weighted", zero_division=0
        )
        rec_ind = recall_score(
            labels, preds, average="weighted", zero_division=0
        )
        try:
            num_classes = probs.shape[1]
            if num_classes == 2:
                auc_ind = roc_auc_score(labels, probs[:, 1])
            else:
                auc_ind = roc_auc_score(labels, probs, multi_class="ovr")
        except ValueError:
            auc_ind = float("nan")

        individual_metrics.append({
            "model_name": m["model_name"],
            "run_id": m["run_id"],
            "f1": f1_ind,
            "accuracy": acc_ind,
            "mcc": mcc_ind,
            "precision": prec_ind,
            "recall": rec_ind,
            "auc": auc_ind,
        })

        # Weighted accumulation
        if pooled_probs is None:
            pooled_probs = m["weight"] * probs
        else:
            pooled_probs = pooled_probs + m["weight"] * probs

    pooled_preds = np.argmax(pooled_probs, axis=1)

    # Ensemble metrics
    acc = accuracy_score(labels_ref, pooled_preds)
    mcc = matthews_corrcoef(labels_ref, pooled_preds)
    precision = precision_score(
        labels_ref, pooled_preds, average="weighted", zero_division=0
    )
    recall = recall_score(
        labels_ref, pooled_preds, average="weighted", zero_division=0
    )
    f1 = f1_score(
        labels_ref, pooled_preds, average="weighted", zero_division=0
    )

    num_classes = pooled_probs.shape[1]
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels_ref, pooled_probs[:, 1])
        else:
            auc = roc_auc_score(labels_ref, pooled_probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    target_labels = list(range(num_classes))
    cm = confusion_matrix(labels_ref, pooled_preds, labels=target_labels)

    far = None
    if mode == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    ensemble_metrics = {
        "accuracy": acc,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "per_class_acc": per_class_acc,
        "far": far,
        "cm": cm,
    }

    return {
        "labels": labels_ref,
        "pooled_preds": pooled_preds,
        "pooled_probs": pooled_probs,
        "ensemble_metrics": ensemble_metrics,
        "individual_metrics": individual_metrics,
    }


# =====================================================================
# 4. Report generation
# =====================================================================

def _axis_labels_for_mode(mode, num_classes, members):
    """Derive human-readable class names from the mode."""
    if mode == "detection":
        return ["background", "target"]
    if mode == "category":
        # Load CLASS_MAP from any member's hyperparameters.json
        for m in members:
            hp_path = m["run_dir"] / "hyperparameters.json"
            if hp_path.exists():
                with open(hp_path) as f:
                    cfg = json.load(f)
                class_map = {int(k): v for k, v in cfg.get("CLASS_MAP", {}).items()}
                return [class_map.get(i, str(i)) for i in range(num_classes)]
    if mode == "instance":
        for m in members:
            hp_path = m["run_dir"] / "hyperparameters.json"
            if hp_path.exists():
                with open(hp_path) as f:
                    cfg = json.load(f)
                itc = cfg.get("INSTANCE_TO_CLASS", {})
                inv = {v: k for k, v in itc.items()}
                return [inv.get(i, str(i)) for i in range(num_classes)]
    return [str(i) for i in range(num_classes)]


def build_report(mode, members, result, output_path):
    """Write a human-readable ensemble evaluation report."""
    em = result["ensemble_metrics"]
    ind = result["individual_metrics"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_ind = max(ind, key=lambda x: x["f1"])
    lift = em["f1"] - best_ind["f1"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ENSEMBLE EVALUATION REPORT\n")
        f.write(f"Mode: {mode} | Timestamp: {ts}\n")
        f.write("=" * 60 + "\n\n")

        # --- Composition ---
        f.write("ENSEMBLE COMPOSITION\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Model':<30} {'Run ID':<20} {'F1':>6}  {'Weight':>7}\n"
        )
        f.write("-" * 60 + "\n")
        for m in sorted(members, key=lambda x: -x["f1"]):
            f.write(
                f"{m['model_name']:<30} {m['run_id']:<20} "
                f"{m['f1']:>6.4f}  {m['weight']:>7.4f}\n"
            )
        f.write("\n")

        # --- Individual model metrics ---
        f.write("INDIVIDUAL MODEL TEST METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Model':<30} {'Acc':>6}  {'F1':>6}  {'MCC':>6}  {'AUC':>6}\n"
        )
        f.write("-" * 60 + "\n")
        for m in sorted(ind, key=lambda x: -x["f1"]):
            auc_str = f"{m['auc']:>6.4f}" if not np.isnan(m["auc"]) else "   N/A"
            f.write(
                f"{m['model_name']:<30} {m['accuracy']:>6.4f}  "
                f"{m['f1']:>6.4f}  {m['mcc']:>6.4f}  {auc_str}\n"
            )
        f.write("\n")

        # --- Ensemble metrics ---
        f.write("ENSEMBLE METRICS (Pooled Confidence)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:    {em['accuracy']:.4f}\n")
        f.write(f"MCC:         {em['mcc']:.4f}\n")
        f.write(f"Precision:   {em['precision']:.4f}\n")
        f.write(f"Recall:      {em['recall']:.4f}\n")
        f.write(f"F1-Score:    {em['f1']:.4f}\n")
        auc_str = (
            f"{em['auc']:.4f}" if not np.isnan(em["auc"]) else "N/A"
        )
        f.write(f"ROC-AUC:     {auc_str}\n")

        if em["far"] is not None:
            f.write(f"False Alarm Rate: {em['far'] * 100:.3f}%\n")

        f.write("\nPer-Class Accuracy:\n")
        num_classes = len(em["per_class_acc"])
        axis_labels = _axis_labels_for_mode(mode, num_classes, members)
        for i, label in enumerate(axis_labels):
            if i < len(em["per_class_acc"]):
                f.write(f"  {label} ({i}): {em['per_class_acc'][i]:.4f}\n")
        f.write("\n")

        # --- Lift ---
        f.write("ENSEMBLE LIFT\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Best Individual:  {best_ind['model_name']} "
            f"(F1: {best_ind['f1']:.4f})\n"
        )
        f.write(f"Ensemble F1:      {em['f1']:.4f}\n")
        sign = "+" if lift >= 0 else ""
        pct = lift / best_ind["f1"] * 100 if best_ind["f1"] > 0 else 0.0
        f.write(f"Lift:             {sign}{lift:.4f} ({sign}{pct:.2f}%)\n")
        f.write("=" * 60 + "\n")

    print(f"  Ensemble report saved: {output_path}")


def save_conf_matrix(mode, members, result, output_path):
    """Save the ensemble confusion matrix as a PNG."""
    em = result["ensemble_metrics"]
    cm = em["cm"]
    num_classes = cm.shape[0]
    axis_labels = _axis_labels_for_mode(mode, num_classes, members)

    fig_size = max(12, num_classes * 1.2)
    annot_size = max(18, min(26, int(240 / num_classes)))

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        annot_kws={"size": annot_size, "weight": "bold"},
        cbar_kws={"shrink": 0.8},
        xticklabels=axis_labels,
        yticklabels=axis_labels,
    )
    plt.title(
        f"Ensemble Confusion Matrix ({mode})", fontsize=26, pad=20
    )
    plt.ylabel("True Label", fontsize=22, labelpad=14)
    plt.xlabel("Predicted Label", fontsize=22, labelpad=14)

    if num_classes > 5:
        plt.xticks(rotation=45, ha="right", fontsize=20)
    else:
        plt.xticks(rotation=0, fontsize=20)

    plt.yticks(rotation=0, fontsize=20)
    plt.gcf().axes[-1].tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Ensemble confusion matrix saved: {output_path}")


# =====================================================================
# 5. Entry point
# =====================================================================

def run_ensemble(mode):
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE — mode: {mode}")
    print(f"{'='*60}")

    members = discover_members(mode)

    if len(members) == 0:
        print(
            f"  [!] No evaluated runs found for mode '{mode}'. "
            "Run eval.py first."
        )
        return

    if len(members) == 1:
        print(
            f"  [!] Only one model found for mode '{mode}'. "
            "Ensemble requires at least two models."
        )
        return

    compute_weights(members)

    print(f"  Members ({len(members)}):")
    for m in sorted(members, key=lambda x: -x["f1"]):
        print(
            f"    {m['model_name']:<30} F1={m['f1']:.4f}  "
            f"weight={m['weight']:.4f}  run={m['run_id']}"
        )

    result = evaluate_ensemble(members, mode)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("saved_models") / "ensemble" / mode / ts
    os.makedirs(out_dir, exist_ok=True)

    report_path = out_dir / "ensemble_report.txt"
    matrix_path = out_dir / "ensemble_conf_matrix.png"

    build_report(mode, members, result, report_path)
    save_conf_matrix(mode, members, result, matrix_path)

    em = result["ensemble_metrics"]
    print(
        f"\n  Ensemble F1: {em['f1']:.4f}  "
        f"Acc: {em['accuracy']:.4f}  "
        f"MCC: {em['mcc']:.4f}"
    )


def main():
    all_modes = ["detection", "category", "instance"]

    # Accept mode as CLI arg or env var, else run all three
    if len(sys.argv) > 1:
        modes = [sys.argv[1]]
    else:
        env_mode = os.environ.get("TRAINING_MODE")
        modes = [env_mode] if env_mode else all_modes

    for mode in modes:
        if mode not in all_modes:
            print(f"[!] Unknown mode '{mode}'. Choose from {all_modes}.")
            sys.exit(1)
        run_ensemble(mode)


if __name__ == "__main__":
    main()
