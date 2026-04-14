"""
Evaluation utilities.

linear_probe_accuracy()  — logistic regression probe per embedding block.
run_full_eval()          — collects embeddings over a DataLoader and computes metrics.
sample_level_eval()      — per-sample inference for diagnostic CSVs.
plot_confusion_matrices() — confusion matrix plots saved to disk.

These are diagnostic tools run periodically during CRL pre-training to
measure embedding quality before downstream head training.
"""

from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def linear_probe_accuracy(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    label_name: str = "label",
) -> dict[str, float]:
    """
    Fit a logistic regression on z_train, evaluate on z_val.
    Returns accuracy and weighted F1.
    Filters out invalid labels (< 0) before fitting.
    """
    valid_train = y_train >= 0
    valid_val   = y_val   >= 0

    if valid_train.sum() < 2 or len(np.unique(y_train[valid_train])) < 2:
        return {f"{label_name}_acc": 0.0, f"{label_name}_f1": 0.0}

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(z_train[valid_train], y_train[valid_train])

    if valid_val.sum() == 0:
        return {f"{label_name}_acc": 0.0, f"{label_name}_f1": 0.0}

    pred = clf.predict(z_val[valid_val])
    acc  = accuracy_score(y_val[valid_val], pred)
    f1   = f1_score(y_val[valid_val], pred, average="weighted", zero_division=0)
    return {f"{label_name}_acc": float(acc), f"{label_name}_f1": float(f1)}


# ---------------------------------------------------------------------------
# Full evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_full_eval(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    primary_sensor: str = "seismic",
    max_batches: int | None = None,
) -> dict:
    """
    Collect embeddings over train and val splits, then compute:
        - Linear probe accuracy for each embedding block (pres, type, inst)
        - Detection AUC using the presence embedding

    max_batches: cap on batches to collect per split (None = full loader).
    Returns a flat metrics dict suitable for logging.
    """
    model.eval()

    def collect(loader):
        e_pres_list, e_type_list = [], []
        vtypes, dets = [], []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[f"x_{primary_sensor}"].to(device)
            ep, et = model.encode_modality(primary_sensor, x)
            e_pres_list.append(ep.cpu().numpy())
            e_type_list.append(et.cpu().numpy())
            vtypes.append(batch["vehicle_type"].numpy())
            dets.append(batch["detection_label"].numpy())
        return (
            np.concatenate(e_pres_list),
            np.concatenate(e_type_list),
            np.concatenate(vtypes),
            np.concatenate(dets),
        )

    ep_tr, et_tr, vt_tr, det_tr = collect(train_loader)
    ep_val, et_val, vt_val, det_val = collect(val_loader)

    metrics = {}

    # Linear probes per embedding block
    metrics.update(linear_probe_accuracy(ep_tr, det_tr, ep_val, det_val, label_name="probe_pres"))
    metrics.update(linear_probe_accuracy(et_tr, vt_tr,  et_val, vt_val,  label_name="probe_type"))

    # Detection AUC: sigmoid of linear probe score on e_pres
    if len(np.unique(det_val)) == 2:
        try:
            clf_det = LogisticRegression(max_iter=500, C=1.0)
            clf_det.fit(ep_tr, det_tr)
            det_scores = clf_det.predict_proba(ep_val)[:, 1]
            auc = roc_auc_score(det_val, det_scores)
            metrics["detection_auc"] = float(auc)
        except Exception:
            metrics["detection_auc"] = 0.0
    else:
        metrics["detection_auc"] = 0.0

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Labels and names
# ---------------------------------------------------------------------------

_INTERV_NAMES = {
    0: "clean",
    1: "white noise",
    2: "brown noise",
    3: "pink noise",
    4: "green noise",
    5: "low-freq osc",
    6: "hi-freq chirp",
    7: "bird chirps",
}

_DET_NAMES  = ["absent", "present"]
_TYPE_NAMES = ["pedestrian", "light", "sport", "utility"]


# ---------------------------------------------------------------------------
# Sample-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_level_eval(
    model,
    loader: DataLoader,
    device: torch.device,
    primary_sensor: str = "seismic",
    max_batches: int | None = None,
) -> pd.DataFrame:
    """
    Run inference over *loader* using downstream det_heads and return a
    DataFrame with one row per sample:

        segment_id  : unique segment identifier
        interv_idx  : 0=clean, 1-7=noise type applied
        true_det    : ground-truth detection label (0/1)
        pred_det    : model detection prediction (0/1)
        true_type   : ground-truth vehicle type (-2/-1=invalid, 0-3=class)
        pred_type   : model type prediction; -1 when true_type invalid

    max_batches: cap on batches to evaluate (None = full loader).
    """
    model.eval()
    rows = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x         = batch[f"x_{primary_sensor}"].to(device)
        seg_ids   = batch["segment_id"].tolist()
        intervs   = batch["interv_idx"].tolist()
        true_det  = batch["detection_label"].tolist()
        true_type = batch["vehicle_type"].tolist()

        e_pres, e_type = model.encode_modality(primary_sensor, x)
        pres_logit, type_logits = model.det_heads[primary_sensor](e_pres, e_type)

        pred_det      = (pres_logit > 0).long().cpu().tolist()
        pred_type_all = type_logits.argmax(dim=1).cpu().tolist()

        for j in range(len(seg_ids)):
            rows.append({
                "segment_id": seg_ids[j],
                "interv_idx": intervs[j],
                "true_det":   true_det[j],
                "pred_det":   pred_det[j],
                "true_type":  true_type[j],
                "pred_type":  pred_type_all[j] if true_type[j] >= 0 else -1,
            })

    model.train()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str = "",
) -> None:
    """
    Generate and save diagnostic plots to *out_dir*:

        1. detection_cm{tag}.png       — binary detection confusion matrix
        2. type_cm{tag}.png            — 4-class vehicle type confusion matrix
        3. det_acc_by_interv{tag}.png  — detection accuracy per intervention type
        4. predictions{tag}.csv        — per-sample predictions
    """
    from sklearn.metrics import confusion_matrix

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    # 1. Detection confusion matrix
    cm_det = confusion_matrix(df["true_det"], df["pred_det"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm_det, annot=True, fmt="d", cmap="Blues",
        xticklabels=_DET_NAMES, yticklabels=_DET_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Assumed (ground truth)")
    ax.set_title("Vehicle Detection")
    fig.tight_layout()
    fig.savefig(out_dir / f"detection_cm{suffix}.png", dpi=150)
    plt.close(fig)

    # 2. Vehicle type confusion matrix (valid labels only)
    type_df = df[df["true_type"] >= 0].copy()
    if not type_df.empty:
        cm_type = confusion_matrix(
            type_df["true_type"], type_df["pred_type"], labels=[0, 1, 2, 3]
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            cm_type, annot=True, fmt="d", cmap="Blues",
            xticklabels=_TYPE_NAMES, yticklabels=_TYPE_NAMES, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Assumed (ground truth)")
        ax.set_title("Vehicle Type Classification")
        fig.tight_layout()
        fig.savefig(out_dir / f"type_cm{suffix}.png", dpi=150)
        plt.close(fig)

    # 3. Detection accuracy per intervention type
    interv_accs = {}
    for idx, grp in df.groupby("interv_idx"):
        if len(grp) == 0:
            continue
        acc = accuracy_score(grp["true_det"], grp["pred_det"])
        label = _INTERV_NAMES.get(int(idx), f"interv_{idx}")
        interv_accs[label] = acc

    if interv_accs:
        labels = list(interv_accs.keys())
        accs   = list(interv_accs.values())
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
        bars = ax.bar(labels, accs, color="steelblue", edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Detection Accuracy")
        ax.set_xlabel("Noise Intervention Type")
        ax.set_title("Detection Accuracy by Intervention")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="chance")
        ax.legend(fontsize=8)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / f"det_acc_by_interv{suffix}.png", dpi=150)
        plt.close(fig)

    # 4. Per-sample CSV
    df_out = df.copy()
    df_out["interv_name"]   = df_out["interv_idx"].map(
        lambda i: _INTERV_NAMES.get(int(i), f"interv_{i}")
    )
    df_out["true_det_name"] = df_out["true_det"].map(
        lambda v: _DET_NAMES[int(v)] if int(v) in (0, 1) else str(v)
    )
    df_out["pred_det_name"] = df_out["pred_det"].map(
        lambda v: _DET_NAMES[int(v)] if int(v) in (0, 1) else str(v)
    )
    df_out["true_type_name"] = df_out["true_type"].map(
        lambda v: _TYPE_NAMES[int(v)] if 0 <= int(v) < len(_TYPE_NAMES) else (
            "background" if int(v) == -1 else "multi" if int(v) == -2 else str(v)
        )
    )
    df_out["pred_type_name"] = df_out["pred_type"].map(
        lambda v: _TYPE_NAMES[int(v)] if 0 <= int(v) < len(_TYPE_NAMES) else "n/a"
    )
    df_out.to_csv(out_dir / f"predictions{suffix}.csv", index=False)
