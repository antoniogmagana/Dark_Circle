"""
Evaluation utilities.

compute_mig()                — Mutual Information Gap (Kim & Mnih 2018).
linear_probe_accuracy()      — logistic regression probe per latent block.
run_full_eval()              — collects z over a DataLoader and computes all metrics.
collapse_metrics()           — detects posterior collapse (active dim fraction, AU).
noise_type_separation_score()— linear separability of noise types in a latent block.

These are diagnostic tools run after CRL pre-training to measure
disentanglement quality before launching downstream head training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mutual_info_score,
)
from sklearn.preprocessing import KBinsDiscretizer


# ---------------------------------------------------------------------------
# MIG score
# ---------------------------------------------------------------------------

def compute_mig(
    z_samples: np.ndarray,
    factor_labels: dict[str, np.ndarray],
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Mutual Information Gap (Kim & Mnih 2018).

    For each ground-truth factor v_k, compute MI with every z_i dimension.
    MIG_k = (MI_top1(v_k) - MI_top2(v_k)) / H(v_k)
    where top1/top2 are the two highest-MI z dimensions for factor k.

    Args:
        z_samples    : (N, d_z) float array of encoder samples
        factor_labels: dict mapping factor name → (N,) integer label array
                       e.g. {"vehicle_type": ..., "detection": ..., "interv_type": ...}
        n_bins       : number of bins for discretising continuous z dims

    Returns dict with per-factor MIG scores and "mean_mig".
    """
    # Discretise z (required for mutual_info_score)
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    z_disc = disc.fit_transform(z_samples).astype(int)  # (N, d_z)

    results = {}
    mig_values = []

    for factor_name, labels in factor_labels.items():
        labels = np.asarray(labels)
        d_z = z_disc.shape[1]

        # MI between each z dim and this factor
        mis = np.array([
            mutual_info_score(labels, z_disc[:, i])
            for i in range(d_z)
        ])
        mis_sorted = np.sort(mis)[::-1]

        if len(mis_sorted) < 2:
            mig_k = 0.0
        else:
            # Entropy of the factor
            counts = np.bincount(labels[labels >= 0])
            probs = counts / counts.sum()
            h_k = -np.sum(probs * np.log(probs + 1e-8))
            if h_k < 1e-8:
                mig_k = 0.0
            else:
                mig_k = float((mis_sorted[0] - mis_sorted[1]) / h_k)

        results[f"mig_{factor_name}"] = mig_k
        mig_values.append(mig_k)

    results["mean_mig"] = float(np.mean(mig_values)) if mig_values else 0.0
    return results


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
    valid_val = y_val >= 0

    if valid_train.sum() < 2 or len(np.unique(y_train[valid_train])) < 2:
        return {f"{label_name}_acc": 0.0, f"{label_name}_f1": 0.0}

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(z_train[valid_train], y_train[valid_train])

    if valid_val.sum() == 0:
        return {f"{label_name}_acc": 0.0, f"{label_name}_f1": 0.0}

    pred = clf.predict(z_val[valid_val])
    acc = accuracy_score(y_val[valid_val], pred)
    f1 = f1_score(
        y_val[valid_val], pred, average="weighted", zero_division=0
    )
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
) -> dict:
    """
    Collect latents over train and val splits, then compute:
        - MIG for vehicle_type, detection, interv_type
        - Linear probe accuracy per latent block
        - Detection AUC

    Returns a flat metrics dict suitable for logging.
    """
    model.eval()

    def collect(loader):
        zs, mus, vtypes, dets, intervs = [], [], [], [], []
        for batch in loader:
            x = batch[f"x_{primary_sensor}"].to(device)
            z, mu, _, _ = model.encode_modality(primary_sensor, x)
            zs.append(z.cpu().numpy())
            mus.append(mu.cpu().numpy())
            vtypes.append(batch["vehicle_type"].numpy())
            dets.append(batch["detection_label"].numpy())
            intervs.append(batch["interv_idx"].numpy())
        return (
            np.concatenate(zs),
            np.concatenate(mus),
            np.concatenate(vtypes),
            np.concatenate(dets),
            np.concatenate(intervs),
        )

    z_tr, mu_tr, vt_tr, det_tr, iv_tr = collect(train_loader)
    z_val, mu_val, vt_val, det_val, iv_val = collect(val_loader)

    metrics = {}

    # --- MIG ---
    mig = compute_mig(
        z_val,
        {
            "vehicle_type": vt_val,
            "detection": det_val,
            "interv_type": iv_val,
        },
    )
    metrics.update(mig)

    # --- Linear probes per latent block ---
    enc = model.encoders[primary_sensor]
    d_z = enc.d_z

    def block(z, sl):
        return z[:, sl]

    # Type block (most diagnostic)
    metrics.update(linear_probe_accuracy(
        block(z_tr, enc.type_idx), vt_tr,
        block(z_val, enc.type_idx), vt_val,
        label_name="probe_type",
    ))
    # Presence block
    metrics.update(linear_probe_accuracy(
        block(z_tr, enc.presence_idx), det_tr,
        block(z_val, enc.presence_idx), det_val,
        label_name="probe_presence",
    ))
    # Full z
    metrics.update(linear_probe_accuracy(
        z_tr, vt_tr, z_val, vt_val, label_name="probe_full_z"
    ))

    # --- Collapse metrics ---
    metrics.update(collapse_metrics(z_val, mu_val))

    # --- Noise type separation ---
    enc = model.encoders[primary_sensor]
    metrics["noise_sep_in_z_noise"] = noise_type_separation_score(
        z_val[:, enc.noise_idx], iv_val
    )
    noise_start = enc.noise_idx.start
    metrics["noise_sep_in_z_veh"] = noise_type_separation_score(
        z_val[:, :noise_start], iv_val
    )

    # --- Detection AUC ---
    # Use presence block sigmoid as detection score
    pres_scores_val = torch.sigmoid(
        torch.from_numpy(block(z_val, enc.presence_idx))
    ).squeeze(-1).numpy()
    if len(np.unique(det_val)) == 2:
        try:
            auc = roc_auc_score(det_val, pres_scores_val)
            metrics["detection_auc"] = float(auc)
        except Exception:
            metrics["detection_auc"] = 0.0
    else:
        metrics["detection_auc"] = 0.0

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Collapse-prevention diagnostics
# ---------------------------------------------------------------------------

def collapse_metrics(
    z: np.ndarray, mu: np.ndarray | None = None
) -> dict[str, float]:
    """
    Measures whether the latent space has collapsed (any dims unused).

    active_dim_frac  : fraction of z dims with std > 0.05.  Target ≥ 0.80.
    spread_ratio     : min_std / max_std.  1.0 = all dims used equally.
    active_units_frac: (Burda et al.) fraction of dims with Var[mu_i] > 0.01.
                       Only computed when mu is provided.

    z  : (N, d_z)
    mu : (N, d_z) optional — posterior mean from the encoder
    """
    stds = z.std(axis=0)
    active_frac = float((stds > 0.05).mean())
    spread_ratio = float(stds.min() / (stds.max() + 1e-8))
    metrics: dict[str, float] = {
        "active_dim_frac": active_frac,
        "spread_ratio": spread_ratio,
    }
    if mu is not None:
        mu_var = mu.var(axis=0)
        metrics["active_units_frac"] = float((mu_var > 0.01).mean())
    return metrics


def noise_type_separation_score(
    z_block: np.ndarray,
    interv_labels: np.ndarray,
) -> float:
    """
    Linear separability of noise types in a latent block.

    Trains a logistic classifier on z_block to predict intervention type
    (0=clean, 1-7=noise type) and returns held-out accuracy.

    High score in z_noise block is DESIRED — noise is localised there.
    Low score in z_veh block is DESIRED — vehicle dims are clean of noise.

    z_block      : (N, d_block)
    interv_labels: (N,) int — 0=clean, 1-7=noise type
    Returns float accuracy.
    """
    valid = interv_labels >= 0
    if valid.sum() < 2 or len(np.unique(interv_labels[valid])) < 2:
        return 0.0
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(z_block[valid], interv_labels[valid])
    return float(accuracy_score(interv_labels[valid], clf.predict(z_block[valid])))
