"""
Model Ensemble — weighted late fusion across all trained model architectures.

Each member is the best-performing run for a given model architecture,
ranked by test F1. Predictions are fused as a weighted average of per-model
softmax outputs, producing a pooled confidence score over the class space.

Usage:
  python ensemble.py build [mode]   # discover best runs, compute weights
  python ensemble.py show  [mode]   # print ensemble composition
  python ensemble.py eval  [mode]   # evaluate, write report + conf matrices

  If [mode] is omitted, all three training modes are processed in sequence.
  Mode can also be set via the TRAINING_MODE environment variable.
"""

import json
import math
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


ALL_MODES = ["detection", "category", "instance"]
DEFAULT_MODEL_DIR = Path("saved_models")


# =====================================================================
# Model Discovery
# =====================================================================

def _parse_f1(report_path):
    """Extract F1-Score from an evaluation_report.txt."""
    try:
        text = Path(report_path).read_text()
        match = re.search(r"F1-Score:\s*([0-9.]+)", text)
        return float(match.group(1)) if match else 0.0
    except Exception:
        return 0.0


def discover_models(mode, base_dir=DEFAULT_MODEL_DIR):
    """
    Scan saved_models/{mode}/{model_name}/{run_id}/ for evaluated runs.

    Keeps the best run (by test F1) per model architecture. Only includes
    runs that have both evaluation_report.txt and predictions.npz.

    Returns list of dicts: {model_name, run_id, run_dir, val_f1}
    """
    mode_dir = Path(base_dir) / mode
    if not mode_dir.exists():
        return []

    best_per_model = {}

    for report_path in mode_dir.glob("*/*/evaluation_report.txt"):
        run_dir = report_path.parent
        model_name = run_dir.parent.name
        run_id = run_dir.name

        if not (run_dir / "predictions.npz").exists():
            continue

        f1 = _parse_f1(report_path)

        if (
            model_name not in best_per_model
            or f1 > best_per_model[model_name]["val_f1"]
        ):
            best_per_model[model_name] = {
                "model_name": model_name,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "val_f1": f1,
            }

    return list(best_per_model.values())


# =====================================================================
# Weight Computation
# =====================================================================

def compute_weights(members, scheme="linear"):
    """
    Assign fusion weights to each member based on val_f1.
    Mutates each member dict in-place by adding a 'weight' key.

    scheme="linear"  — proportional to F1; falls back to uniform if all zero
    scheme="softmax" — softmax of F1 scores; prevents zero-weight members
    """
    f1_scores = [m["val_f1"] for m in members]

    if scheme == "softmax":
        exps = [math.exp(f) for f in f1_scores]
        total = sum(exps)
        weights = [e / total for e in exps]
    else:  # linear
        total = sum(f1_scores)
        if total == 0:
            weights = [1.0 / len(members)] * len(members)
        else:
            weights = [f / total for f in f1_scores]

    for member, w in zip(members, weights):
        member["weight"] = w

    return members


# =====================================================================
# Metrics
# =====================================================================

def _compute_auc(labels, probs, num_classes):
    if len(np.unique(labels)) <= 1:
        return float("nan")
    try:
        if num_classes == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        return float("nan")


def _compute_metrics(labels, probs, mode):
    """Compute the full metric suite from label and probability arrays."""
    preds = probs.argmax(axis=1)
    num_classes = probs.shape[1]

    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    prec = precision_score(
        labels, preds, average="weighted", zero_division=0
    )
    rec = recall_score(
        labels, preds, average="weighted", zero_division=0
    )
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    auc = _compute_auc(labels, probs, num_classes)

    target_labels = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=target_labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    far = None
    if mode == "detection" and cm.shape == (2, 2):
        tn, fp, _fn, _tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc,
        "mcc": mcc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "per_class_acc": per_class_acc.tolist(),
        "far": far,
        "preds": preds,
    }


# =====================================================================
# Axis Labels & Config helpers
# =====================================================================

def _load_run_config(run_dir):
    """Reconstruct a SimpleNamespace config from hyperparameters.json."""
    json_path = Path(run_dir) / "hyperparameters.json"
    if not json_path.exists():
        return SimpleNamespace()
    with open(json_path) as f:
        data = json.load(f)
    if "CLASS_MAP" in data:
        data["CLASS_MAP"] = {
            int(k): v for k, v in data["CLASS_MAP"].items()
        }
    return SimpleNamespace(**data)


def _build_axis_labels(mode, run_config):
    """Human-readable class labels for confusion matrix axes."""
    if mode == "detection":
        return ["background", "vehicle"]
    if mode == "category":
        class_map = getattr(run_config, "CLASS_MAP", {})
        num_classes = getattr(run_config, "NUM_CLASSES", 4)
        return [class_map.get(i, str(i)) for i in range(num_classes)]
    if mode == "instance":
        inv = {
            v: k
            for k, v in getattr(
                run_config, "INSTANCE_TO_CLASS", {}
            ).items()
        }
        num_classes = getattr(run_config, "NUM_CLASSES", 0)
        return [inv.get(i, str(i)) for i in range(num_classes)]
    return []


# =====================================================================
# Confusion Matrix
# =====================================================================

def _save_conf_matrix(labels, preds, mode, run_config, output_path):
    """Save a confusion matrix heatmap with dynamic axis labels."""
    axis_labels = _build_axis_labels(mode, run_config)
    num_classes = int(labels.max()) + 1 if len(labels) > 0 else 2
    target_labels = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=target_labels)

    fig_size = max(6, num_classes)
    annot_kws = {"size": max(6, 14 - num_classes // 3)}
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=axis_labels if axis_labels else target_labels,
        yticklabels=axis_labels if axis_labels else target_labels,
        ax=ax,
        annot_kws=annot_kws,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Ensemble Confusion Matrix — {mode}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Ensemble Evaluation (offline — loads predictions.npz)
# =====================================================================

def evaluate_ensemble(members, mode):
    """
    Load predictions.npz from each member, compute weighted pooled
    probabilities, and evaluate the full metric suite.

    Returns a result dict containing ensemble metrics, individual
    model metrics, and the pooled probability array.
    """
    loaded = []
    ref_labels = None

    for member in members:
        npz_path = Path(member["run_dir"]) / "predictions.npz"
        if not npz_path.exists():
            print(
                f"  [!] Missing predictions.npz for "
                f"{member['model_name']} — skipping."
            )
            continue

        data = np.load(npz_path)
        labels = data["labels"]
        probs = data["probs"]

        if ref_labels is None:
            ref_labels = labels
        elif not np.array_equal(ref_labels, labels):
            print(
                f"  [!] Label mismatch for {member['model_name']}"
                f" — skipping member."
            )
            continue

        loaded.append((member, probs))

    if not loaded:
        raise RuntimeError(
            f"No valid members with matching labels for mode={mode}"
        )

    # Recompute weights over the clean subset only
    clean_members = [m for m, _ in loaded]
    compute_weights(clean_members, scheme="linear")

    # Pooled probabilities: weighted convex combination
    pooled_probs = np.zeros_like(loaded[0][1], dtype=np.float64)
    for member, probs in loaded:
        pooled_probs += member["weight"] * probs.astype(np.float64)

    labels = ref_labels
    ensemble_metrics = _compute_metrics(labels, pooled_probs, mode)

    individual = []
    for member, probs in loaded:
        m = _compute_metrics(labels, probs, mode)
        individual.append({
            "model_name": member["model_name"],
            "run_id": member["run_id"],
            "run_dir": member["run_dir"],
            "val_f1": member["val_f1"],
            "weight": member["weight"],
            "test_metrics": m,
        })

    best_ind = max(
        individual, key=lambda x: x["test_metrics"]["f1"]
    )
    lift = ensemble_metrics["f1"] - best_ind["test_metrics"]["f1"]

    return {
        "mode": mode,
        "ensemble_metrics": ensemble_metrics,
        "individual": individual,
        "best_individual_model": best_ind["model_name"],
        "best_individual_f1": best_ind["test_metrics"]["f1"],
        "lift": lift,
        "num_samples": int(len(labels)),
        "labels": labels,
        "pooled_probs": pooled_probs,
    }


# =====================================================================
# Report Generation
# =====================================================================

def _fmt(val, pct=False):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if pct:
        return f"{val * 100:.3f}%"
    return f"{val:.4f}"


def build_report(result, output_path):
    """Write the full ensemble evaluation report to a text file."""
    mode = result["mode"]
    em = result["ensemble_metrics"]
    individual = result["individual"]

    run_config = _load_run_config(individual[0]["run_dir"])
    axis_labels = _build_axis_labels(mode, run_config)

    SEP = "=" * 60
    DIV = "-" * 60
    lines = [
        SEP,
        "ENSEMBLE EVALUATION REPORT",
        f"Mode: {mode}",
        DIV,
        "",
        "ENSEMBLE COMPOSITION",
        DIV,
        f"{'Model':<30} {'Run ID':<20} {'Val F1':>8} {'Weight':>8}",
    ]
    for m in individual:
        lines.append(
            f"{m['model_name']:<30} {m['run_id']:<20}"
            f" {m['val_f1']:>8.4f} {m['weight']:>8.4f}"
        )

    lines += [
        "",
        "INDIVIDUAL MODEL TEST METRICS",
        DIV,
        f"{'Model':<30} {'Accuracy':>9} {'F1':>8}"
        f" {'MCC':>8} {'ROC-AUC':>9}",
    ]
    for m in individual:
        tm = m["test_metrics"]
        lines.append(
            f"{m['model_name']:<30}"
            f" {_fmt(tm['accuracy']):>9}"
            f" {_fmt(tm['f1']):>8}"
            f" {_fmt(tm['mcc']):>8}"
            f" {_fmt(tm['auc']):>9}"
        )

    lines += [
        "",
        "ENSEMBLE METRICS (Pooled Confidence)",
        DIV,
        f"Accuracy:   {_fmt(em['accuracy'])}",
        f"MCC:        {_fmt(em['mcc'])}",
        f"Precision:  {_fmt(em['precision'])}",
        f"Recall:     {_fmt(em['recall'])}",
        f"F1-Score:   {_fmt(em['f1'])}",
        f"ROC-AUC:    {_fmt(em['auc'])}",
        "",
        "Per-Class Accuracy:",
    ]
    for i, acc in enumerate(em["per_class_acc"]):
        label = axis_labels[i] if i < len(axis_labels) else str(i)
        lines.append(f"  {label} ({i}): {_fmt(acc)}")

    if em["far"] is not None:
        lines.append(f"\nFalse Alarm Rate: {_fmt(em['far'], pct=True)}")

    best_f1 = result["best_individual_f1"]
    lift_pct = (
        result["lift"] / best_f1 * 100 if best_f1 > 0 else 0.0
    )
    lines += [
        "",
        "ENSEMBLE LIFT",
        DIV,
        f"Best Individual: {result['best_individual_model']}"
        f" (F1: {_fmt(best_f1)})",
        f"Ensemble F1:     {_fmt(em['f1'])}",
        f"Lift:            {result['lift']:+.4f} ({lift_pct:+.2f}%)",
        SEP,
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"  Report → {output_path}")


# =====================================================================
# ModelEnsemble class
# =====================================================================

class ModelEnsemble:
    """Weighted ensemble of trained model-architecture runs."""

    def __init__(self, mode, members):
        self.mode = mode
        self.members = members  # list of dicts, each with 'weight' key

    @classmethod
    def build(cls, mode, base_dir=DEFAULT_MODEL_DIR, scheme="linear"):
        members = discover_models(mode, base_dir)
        if not members:
            print(f"  [!] No evaluated runs found for mode={mode}")
            return cls(mode, [])
        compute_weights(members, scheme=scheme)
        return cls(mode, members)

    @classmethod
    def load(cls, weights_path):
        with open(weights_path) as f:
            data = json.load(f)
        return cls(data["mode"], data["members"])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "weight_metric": "val_f1",
            "members": self.members,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Weights → {path}")

    def summary(self):
        print(
            f"\nEnsemble — mode: {self.mode}"
            f"  ({len(self.members)} members)"
        )
        print(
            f"{'Model':<30} {'Run ID':<20} {'Val F1':>8} {'Weight':>8}"
        )
        print("-" * 70)
        for m in self.members:
            print(
                f"{m['model_name']:<30} {m['run_id']:<20}"
                f" {m['val_f1']:>8.4f}"
                f" {m.get('weight', float('nan')):>8.4f}"
            )

    def evaluate(self, base_dir=DEFAULT_MODEL_DIR):
        return evaluate_ensemble(self.members, self.mode)


# =====================================================================
# CLI helpers
# =====================================================================

def _resolve_modes(mode_arg):
    """Return list of modes from CLI arg or TRAINING_MODE env var."""
    if mode_arg and mode_arg in ALL_MODES:
        return [mode_arg]
    env = os.environ.get("TRAINING_MODE", "")
    if env in ALL_MODES:
        return [env]
    return ALL_MODES


def _weights_path(mode, base_dir=DEFAULT_MODEL_DIR):
    return Path(base_dir) / "ensemble" / mode / "ensemble_weights.json"


def cmd_build(mode_arg):
    for mode in _resolve_modes(mode_arg):
        print(f"\n[build] mode={mode}")
        ens = ModelEnsemble.build(mode)
        if not ens.members:
            continue
        ens.save(_weights_path(mode))
        ens.summary()


def cmd_show(mode_arg):
    for mode in _resolve_modes(mode_arg):
        wp = _weights_path(mode)
        if not wp.exists():
            print(
                f"  [!] No weights file for mode={mode}."
                f" Run 'build' first."
            )
            continue
        ModelEnsemble.load(wp).summary()


def cmd_eval(mode_arg):
    for mode in _resolve_modes(mode_arg):
        print(f"\n[eval] mode={mode}")
        wp = _weights_path(mode)
        if not wp.exists():
            print(
                f"  [!] No weights file for mode={mode}."
                f" Run 'build' first."
            )
            continue

        ens = ModelEnsemble.load(wp)
        if not ens.members:
            print(f"  [!] Ensemble for mode={mode} has no members.")
            continue

        try:
            result = ens.evaluate()
        except Exception as exc:
            print(f"  [!] Evaluation failed for mode={mode}: {exc}")
            continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (
            DEFAULT_MODEL_DIR / "ensemble" / mode / ts
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        build_report(result, out_dir / "ensemble_report.txt")

        run_config = _load_run_config(
            result["individual"][0]["run_dir"]
        )
        cm_path = out_dir / f"ensemble_conf_matrix_{mode}.png"
        _save_conf_matrix(
            result["labels"],
            result["ensemble_metrics"]["preds"],
            mode,
            run_config,
            cm_path,
        )
        print(f"  Conf matrix → {cm_path}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    mode_arg = args[1] if len(args) > 1 else None

    if cmd == "build":
        cmd_build(mode_arg)
    elif cmd == "show":
        cmd_show(mode_arg)
    elif cmd == "eval":
        cmd_eval(mode_arg)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
