#!/usr/bin/env python3
"""
validate_run.py — Post-run validation (README §8).

Loads a trained CRL checkpoint and runs a structured battery of checks:
  §8.1  Unit-level integrity   (shapes, finiteness, acyclicity, filterbank bounds)
  §8.2  Representation quality (MIG, linear probes, cross-block interference)
  §8.3  Causal structure       (SCM adjacency, intervention localization, counterfactual)
  §8.4  Downstream performance (AUC, accuracy, F1; Phase 1 probe vs Phase 2 head)
  §8.6  Scientific integrity   (checkpoint metric stationarity from crl_metrics.csv)

Exits 0 if all hard checks pass, 1 if any FAIL.

Usage:
    python validate_run.py
    python validate_run.py --save-dir ./saved_crl \\
                           --data-dir ../data_files/parsed/train \\
                           --val-dir  ../data_files/parsed/val
    python validate_run.py --checkpoint crl_final.pth
    python validate_run.py --device cpu --batch-size 32
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import SensorDataset, collate_single
from crl_vehicle.data.transforms import apply_intervention, N_INTERVENTIONS
from training.eval import compute_mig, linear_probe_accuracy, collapse_metrics, noise_type_separation_score
from training.trainer import CRLModel


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
INFO = "INFO"

_results: list[tuple[str, str, str]] = []


def _header(msg: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {msg}")
    print("=" * 65)


def _result(status: str, name: str, detail: str = "") -> None:
    symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "INFO": "·"}[status]
    print(f"  {symbol} [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append((status, name, detail))


def _check(condition: bool, name: str, detail: str = "") -> bool:
    _result(PASS if condition else FAIL, name, detail)
    return condition


def _check_above(value: float, threshold: float, name: str, fmt: str = ".4f") -> bool:
    ok = value >= threshold
    _result(PASS if ok else FAIL, name, f"{value:{fmt}} (need ≥ {threshold:{fmt}})")
    return ok


def _check_below(value: float, threshold: float, name: str, fmt: str = ".4f") -> bool:
    ok = value <= threshold
    _result(PASS if ok else FAIL, name, f"{value:{fmt}} (need ≤ {threshold:{fmt}})")
    return ok


# ---------------------------------------------------------------------------
# Latent collection (shared across §8.2 and §8.3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_latents(
    model: CRLModel,
    loader,
    device: torch.device,
    sensor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, vehicle_type, detection_label, interv_idx) numpy arrays over a loader."""
    zs, vtypes, dets, intervs = [], [], [], []
    for batch in loader:
        x = batch[f"x_{sensor}"].to(device)
        z, _, _, _ = model.encode_modality(sensor, x)
        zs.append(z.cpu().numpy())
        vtypes.append(batch["vehicle_type"].numpy())
        dets.append(batch["detection_label"].numpy())
        intervs.append(batch["interv_idx"].numpy())
    return (
        np.concatenate(zs),
        np.concatenate(vtypes),
        np.concatenate(dets),
        np.concatenate(intervs),
    )


# ---------------------------------------------------------------------------
# §8.1  Unit-level checks
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_unit_level(
    model: CRLModel,
    batch: dict,
    device: torch.device,
    cfg: CRLConfig,
) -> None:
    _header("§8.1  Unit-Level Checks")

    all_shapes_ok = True
    all_finite = True

    for sensor in model.sensors:
        x = batch[f"x_{sensor}"].to(device)
        B = x.shape[0]
        mod = cfg.modality_cfg(sensor)

        y = model.filterbanks[sensor](x)
        exp_fb = (B, mod.filterbank_out_channels, mod.t_prime)
        if tuple(y.shape) != exp_fb:
            _result(FAIL, f"filterbank shape [{sensor}]",
                    f"got {tuple(y.shape)}, expected {exp_fb}")
            all_shapes_ok = False
        if not y.isfinite().all():
            _result(FAIL, f"filterbank finite [{sensor}]", "non-finite values detected")
            all_finite = False

        h = model.ssms[sensor](y)
        exp_ssm = (B, mod.t_prime, cfg.d_model)
        if tuple(h.shape) != exp_ssm:
            _result(FAIL, f"SSM shape [{sensor}]",
                    f"got {tuple(h.shape)}, expected {exp_ssm}")
            all_shapes_ok = False

        z, _, _ = model.encoders[sensor](h)
        if tuple(z.shape) != (B, cfg.d_z):
            _result(FAIL, f"encoder z shape [{sensor}]",
                    f"got {tuple(z.shape)}, expected ({B}, {cfg.d_z})")
            all_shapes_ok = False
        if not z.isfinite().all():
            _result(FAIL, f"encoder z finite [{sensor}]", "non-finite values")
            all_finite = False

        # Filterbank frequency bounds
        freqs = model.filterbanks[sensor].center_frequencies().cpu().numpy()
        in_bounds = (freqs >= mod.f_min).all() and (freqs <= mod.f_max).all()
        _result(
            PASS if in_bounds else WARN,
            f"filterbank frequencies in bounds [{sensor}]",
            f"range [{freqs.min():.1f}, {freqs.max():.1f}] Hz "
            f"(config [{mod.f_min}, {mod.f_max}] Hz)",
        )

    if all_shapes_ok:
        _result(PASS, "shape trace: Filterbank → SSM → Encoder all correct")
    if all_finite:
        _result(PASS, "all intermediate tensors are finite")

    # Modality sample-rate assertions
    _check(
        cfg.audio_cfg.sample_rate == 16000,
        "audio sample rate == 16000 Hz",
        f"got {cfg.audio_cfg.sample_rate} Hz",
    )
    _check(
        cfg.audio_cfg.f_max == 7500.0,
        "audio f_max == 7500 Hz (engine harmonics preserved)",
        f"got {cfg.audio_cfg.f_max} Hz",
    )
    _check(
        cfg.seismic_cfg.sample_rate == 200,
        "seismic sample rate == 200 Hz",
        f"got {cfg.seismic_cfg.sample_rate} Hz",
    )

    # SCM acyclicity at convergence
    acyc = model.scm.acyclicity_loss().item()
    _check_below(acyc, 0.1, "acyclicity_loss at convergence", fmt=".5f")
    _result(INFO, "acyclicity_loss raw value", f"{acyc:.5f}")


# ---------------------------------------------------------------------------
# §8.2  Representation quality
# ---------------------------------------------------------------------------

def check_representation_quality(
    model: CRLModel,
    z_tr: np.ndarray,
    vt_tr: np.ndarray,
    det_tr: np.ndarray,
    iv_tr: np.ndarray,
    z_val: np.ndarray,
    vt_val: np.ndarray,
    det_val: np.ndarray,
    iv_val: np.ndarray,
    cfg: CRLConfig,
    primary_sensor: str,
) -> dict:
    _header("§8.2  Representation Quality")
    metrics: dict = {}

    enc = model.encoders[primary_sensor]

    def blk(z: np.ndarray, sl: slice) -> np.ndarray:
        return z[:, sl]

    # --- MIG ---
    mig = compute_mig(
        z_val,
        {"vehicle_type": vt_val, "detection": det_val, "interv_type": iv_val},
    )
    metrics.update(mig)

    _check_above(mig["mig_vehicle_type"], 0.2,
                 "MIG [vehicle_type] > 0.2 (z_type encodes vehicle class)")
    _check_above(mig["mig_detection"],    0.2,
                 "MIG [detection] > 0.2 (z_presence encodes vehicle presence)")
    _check_above(mig["mig_interv_type"],  0.2,
                 "MIG [interv_type] > 0.2 (interventions localized to z_noise)")
    _result(INFO, "mean MIG", f"{mig['mean_mig']:.4f}")

    # --- Linear probes per semantic block ---
    m_type = linear_probe_accuracy(
        blk(z_tr, enc.type_idx), vt_tr,
        blk(z_val, enc.type_idx), vt_val,
        label_name="probe_type",
    )
    metrics.update(m_type)
    _check_above(m_type["probe_type_acc"], 0.25,
                 "probe z_type accuracy > 25% (4-class random baseline)")
    _result(INFO, "probe z_type F1", f"{m_type['probe_type_f1']:.4f}")

    m_pres = linear_probe_accuracy(
        blk(z_tr, enc.presence_idx), det_tr,
        blk(z_val, enc.presence_idx), det_val,
        label_name="probe_presence",
    )
    metrics.update(m_pres)
    _result(INFO, "probe z_presence accuracy", f"{m_pres['probe_presence_acc']:.4f}")
    _result(INFO, "probe z_presence F1",       f"{m_pres['probe_presence_f1']:.4f}")

    # Detection AUC: sigmoid of z_presence as score
    pres_scores = torch.sigmoid(
        torch.from_numpy(blk(z_val, enc.presence_idx))
    ).squeeze(-1).numpy()
    valid = det_val >= 0
    if valid.any() and len(np.unique(det_val[valid])) == 2:
        try:
            auc = float(roc_auc_score(det_val[valid], pres_scores[valid]))
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
    metrics["detection_auc"] = auc
    _check_above(auc, 0.5, "detection AUC (z_presence sigmoid) > 0.5 majority-class baseline")

    # --- Cross-block interference tests ---
    # type probe on z_presence (should be near chance — 25%)
    xb1 = linear_probe_accuracy(
        blk(z_tr, enc.presence_idx), vt_tr,
        blk(z_val, enc.presence_idx), vt_val,
        label_name="xblock_pres_type",
    )
    xb1_acc = xb1.get("xblock_pres_type_acc", 0.0)
    metrics["xblock_pres_type_acc"] = xb1_acc
    _result(
        PASS if xb1_acc < 0.40 else FAIL,
        "cross-block: type probe on z_presence should be near chance",
        f"acc={xb1_acc:.4f} (fail if > 0.40 — z_presence leaks vehicle-type info)",
    )

    # detection probe on z_noise (should be near chance — ~50%)
    xb2 = linear_probe_accuracy(
        blk(z_tr, enc.noise_idx), det_tr,
        blk(z_val, enc.noise_idx), det_val,
        label_name="xblock_noise_det",
    )
    xb2_acc = xb2.get("xblock_noise_det_acc", 0.0)
    metrics["xblock_noise_det_acc"] = xb2_acc
    _result(
        PASS if xb2_acc < 0.65 else FAIL,
        "cross-block: detection probe on z_noise should be near chance",
        f"acc={xb2_acc:.4f} (fail if > 0.65 — z_noise leaks detection signal)",
    )

    # --- Collapse-prevention checks ---
    cm = collapse_metrics(z_val, mu=None)
    metrics.update(cm)
    _check_above(
        cm["active_dim_frac"], 0.80,
        "active_dim_frac > 80% (< 20% of latent dims collapsed)",
    )
    _result(INFO, "spread_ratio (min_std/max_std)", f"{cm['spread_ratio']:.4f}")

    # --- Noise localisation checks (core scientific requirement) ---
    # Noise types should be linearly separable in z_noise (localised there)
    # but NOT in z_veh (vehicle dims should be clean of noise labels).
    noise_sep_noise = noise_type_separation_score(blk(z_val, enc.noise_idx), iv_val)
    noise_sep_veh   = noise_type_separation_score(blk(z_val, slice(0, enc.noise_idx.start)), iv_val)
    metrics["noise_sep_in_z_noise"] = noise_sep_noise
    metrics["noise_sep_in_z_veh"]   = noise_sep_veh
    _check_above(
        noise_sep_noise, 0.30,
        "noise types separable in z_noise (> 30%) — noise is localised",
    )
    _check_below(
        noise_sep_veh, 0.20,
        "noise types NOT separable in z_veh (< 20%) — vehicle dims clean of noise",
    )

    return metrics


# ---------------------------------------------------------------------------
# §8.3  Causal structure validity
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_causal_structure(
    model: CRLModel,
    batch: dict,
    device: torch.device,
    cfg: CRLConfig,
) -> None:
    _header("§8.3  Causal Structure Validity")

    # SCM adjacency matrix printout
    A = model.scm.adjacency().cpu().numpy()  # (d_z, d_z)
    labels = (
        ["pres"] * cfg.d_z_presence
        + [f"ty{i}" for i in range(cfg.d_z_type)]
        + ["prox"] * cfg.d_z_proximity
        + [f"ns{i}" for i in range(cfg.d_z_noise)]
    )
    _result(INFO, "SCM learned adjacency (entry [i,j] = weight of edge i→j)", "")
    hdr = "       " + "  ".join(f"{l:>4}" for l in labels)
    print(f"    {hdr}")
    for i, lbl in enumerate(labels):
        row = "  ".join(f"{A[i, j]:4.2f}" for j in range(cfg.d_z))
        print(f"    {lbl:>4}: {row}")

    # Acyclicity near zero
    acyc = model.scm.acyclicity_loss().item()
    _check_below(acyc, 0.05, "acyclicity_loss near zero at convergence", fmt=".5f")

    # Intervention mask correctness:
    # known interventions should activate only noise dims (never presence/type).
    noise_start = cfg.d_z_presence + cfg.d_z_type + cfg.d_z_proximity
    interv_ok = True
    for k in range(1, min(N_INTERVENTIONS + 1, 5)):
        idx = torch.full(
            (batch["interv_idx"].shape[0],), k, dtype=torch.long, device=device
        )
        mask = model.known_interv(idx)  # (B, d_z)
        pres_lit  = mask[:, :cfg.d_z_presence].any().item()
        type_lit  = mask[:, cfg.d_z_presence : cfg.d_z_presence + cfg.d_z_type].any().item()
        noise_lit = mask[:, noise_start:].any().item()
        if pres_lit or type_lit or not noise_lit:
            _result(FAIL, f"intervention mask [k={k}]",
                    f"pres={pres_lit}, type={type_lit}, noise={noise_lit}")
            interv_ok = False
    if interv_ok:
        _result(PASS, "intervention masks activate only z_noise (not z_presence or z_type)")

    # Intervention effect localization:
    # apply each known noise type to a batch; the z_noise centroid should shift
    # across intervention groups, while z_type centroid should remain stable.
    sensor  = "seismic" if "seismic" in model.sensors else model.sensors[0]
    mod_cfg = cfg.modality_cfg(sensor)
    enc     = model.encoders[sensor]
    x_orig  = batch[f"x_{sensor}"]  # (B, C, W) CPU

    z_noise_centroids: list[np.ndarray] = []
    z_type_centroids:  list[np.ndarray] = []
    for k in range(N_INTERVENTIONS + 1):
        x_k = torch.stack([
            apply_intervention(x_orig[i], k, mod_cfg.sample_rate)
            for i in range(x_orig.shape[0])
        ]).to(device)
        z, _, _ = model.encode_modality(sensor, x_k)
        z_noise_centroids.append(z[:, enc.noise_idx].mean(0).cpu().numpy())
        z_type_centroids.append(z[:, enc.type_idx].mean(0).cpu().numpy())

    noise_spread = float(np.std(np.stack(z_noise_centroids)))
    type_spread  = float(np.std(np.stack(z_type_centroids)))
    _result(
        PASS if noise_spread > type_spread else WARN,
        "intervention effect localized: z_noise centroid spread > z_type spread",
        f"z_noise spread={noise_spread:.4f}  z_type spread={type_spread:.4f}",
    )

    # Counterfactual consistency:
    # for a single fixed input, manually set z_type to each of the 4 class
    # directions and check that the decoder produces qualitatively different
    # spectral envelopes (mean pairwise MSE should be > 0).
    z_base, _, _, _ = model.encode_modality(sensor, x_orig[:1].to(device))
    decoder     = model.decoders[sensor]
    type_start  = cfg.d_z_presence
    type_end    = type_start + cfg.d_z_type

    cf_recs: list[np.ndarray] = []
    for cls_idx in range(cfg.d_z_type):
        z_cf = z_base.clone()
        z_cf[0, type_start:type_end] = -3.0          # suppress all type dims
        z_cf[0, type_start + cls_idx] = 3.0           # activate one class
        cf_recs.append(decoder(z_cf).squeeze(0).cpu().numpy())

    pairwise_mse = [
        float(np.mean((cf_recs[i] - cf_recs[j]) ** 2))
        for i in range(len(cf_recs))
        for j in range(i + 1, len(cf_recs))
    ]
    mean_cf_mse = float(np.mean(pairwise_mse))
    _result(
        PASS if mean_cf_mse > 1e-6 else FAIL,
        "counterfactual: different z_type inputs → different decoder outputs",
        f"mean pairwise MSE = {mean_cf_mse:.6f}",
    )


# ---------------------------------------------------------------------------
# §8.4  Downstream task performance
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_downstream_performance(
    model: CRLModel,
    val_loader,
    device: torch.device,
    save_dir: Path,
    primary_sensor: str,
    eval_metrics: dict,
) -> None:
    _header("§8.4  Downstream Task Performance")

    probe_type_acc = eval_metrics.get("probe_type_acc", 0.0)
    probe_type_f1  = eval_metrics.get("probe_type_f1",  0.0)
    det_auc        = eval_metrics.get("detection_auc",  0.0)
    probe_pres_f1  = eval_metrics.get("probe_presence_f1", 0.0)

    print("\n  Phase 1 — Linear probe on frozen z (backbone evaluation):")
    _check_above(det_auc,        0.5,  "  detection AUC > 0.5 (majority-class baseline)")
    _check_above(probe_type_acc, 0.25, "  vehicle type accuracy > 25% (4-class random baseline)")
    _result(INFO, "  probe presence F1", f"{probe_pres_f1:.4f}")
    _result(INFO, "  probe type F1",     f"{probe_type_f1:.4f}")

    # Phase 2: fine-tuned detection/classification head
    head_ckpt = save_dir / f"det_head_{primary_sensor}_best.pth"
    if not head_ckpt.exists():
        _result(WARN, "Phase 2 head checkpoint not found",
                f"expected {head_ckpt} — run --phase downstream first")
        return

    model.det_heads[primary_sensor].load_state_dict(
        torch.load(head_ckpt, map_location=device, weights_only=True)
    )

    det_true, det_pred, det_scores = [], [], []
    cls_true, cls_pred = [], []
    enc = model.encoders[primary_sensor]

    for batch in val_loader:
        x     = batch[f"x_{primary_sensor}"].to(device)
        vtype = batch["vehicle_type"]
        det   = batch["detection_label"]

        z, _, _, _      = model.encode_modality(primary_sensor, x)
        z_pres, z_type_r, _, _ = enc.split_z_raw(z)
        pres_logit, type_logits = model.det_heads[primary_sensor](z_pres, z_type_r)

        scores = torch.sigmoid(pres_logit.squeeze(-1)).cpu().numpy()
        det_scores.extend(scores.tolist())
        det_pred.extend((pres_logit.squeeze(-1) > 0).long().cpu().tolist())
        det_true.extend(det.tolist())

        cls_mask = vtype >= 0
        if cls_mask.any():
            cls_pred.extend(type_logits[cls_mask.to(device)].argmax(1).cpu().tolist())
            cls_true.extend(vtype[cls_mask].tolist())

    print("\n  Phase 2 — Fine-tuned detection/classification head:")
    if len(np.unique(det_true)) == 2:
        auc2    = float(roc_auc_score(det_true, det_scores))
        det_f1_2 = float(f1_score(det_true, det_pred, average="weighted", zero_division=0))
        _check_above(auc2, 0.5, "  detection AUC > 0.5 (Phase 2)")
        _result(INFO, "  detection weighted F1 (Phase 2)", f"{det_f1_2:.4f}")
    else:
        auc2 = 0.0
        _result(WARN, "  only one detection class in val split — AUC undefined")

    if cls_true:
        cls_acc_2 = float(accuracy_score(cls_true, cls_pred))
        cls_f1_2  = float(f1_score(cls_true, cls_pred, average="weighted", zero_division=0))
        _check_above(cls_acc_2, 0.25, "  vehicle type accuracy > 25% random (Phase 2)")
        _result(INFO, "  vehicle type F1 (Phase 2)", f"{cls_f1_2:.4f}")

        print("\n  Phase 1 vs Phase 2 (probe vs fine-tuned head):")
        _result(INFO, "  type accuracy",
                f"probe={probe_type_acc:.4f}  head={cls_acc_2:.4f}  "
                f"Δ={cls_acc_2 - probe_type_acc:+.4f}")
        _result(
            PASS if cls_acc_2 >= probe_type_acc else WARN,
            "  fine-tuning adds value beyond linear decodability",
            "(warn: head < probe — may need more backbone pre-training)",
        )


# ---------------------------------------------------------------------------
# §8.6  Scientific integrity checks
# ---------------------------------------------------------------------------

def check_scientific_integrity(save_dir: Path, cfg: CRLConfig) -> None:
    _header("§8.6  Scientific Integrity Checks")

    metrics_path = save_dir / "crl_metrics.csv"
    if not metrics_path.exists():
        _result(WARN, "crl_metrics.csv not found",
                f"expected at {metrics_path}; run --phase crl first")
        return

    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        _result(WARN, "crl_metrics.csv is empty")
        return

    _result(INFO, f"training history: {len(rows)} epochs logged")

    has_val_total = "val_total" in rows[0]
    has_val_ckpt  = "val_ckpt"  in rows[0]
    _check(
        has_val_total and has_val_ckpt,
        "crl_metrics.csv has both val_total (annealing) and val_ckpt (fixed-beta)",
        f"val_total={has_val_total}, val_ckpt={has_val_ckpt}",
    )
    if not (has_val_total and has_val_ckpt):
        return

    val_total = [float(r["val_total"]) for r in rows if r.get("val_total")]
    val_ckpt  = [float(r["val_ckpt"])  for r in rows if r.get("val_ckpt")]

    # val_ckpt should decrease post-annealing (epoch-invariant metric improves)
    post_start = cfg.beta_anneal_epochs
    post_ckpt  = val_ckpt[post_start:] if len(val_ckpt) > post_start else val_ckpt
    if len(post_ckpt) >= 2:
        improves = min(post_ckpt) < post_ckpt[0]
        _result(
            PASS if improves else WARN,
            "val_ckpt (fixed-beta) decreases post-annealing",
            f"initial={post_ckpt[0]:.4f}  min={min(post_ckpt):.4f}",
        )

    # val_total and val_ckpt should diverge during annealing (non-stationarity visible)
    if len(val_total) == len(val_ckpt):
        max_div = max(abs(a - b) for a, b in zip(val_total, val_ckpt))
        _result(INFO, "max divergence val_total vs val_ckpt", f"{max_div:.4f}")
        _result(
            PASS if max_div > 1e-6 else WARN,
            "val_total and val_ckpt diverge (confirms beta annealing is active)",
            "(warn: identical — annealing may have had no effect)",
        )

    # Beta schedule reaches beta_end
    if rows[0].get("beta") is not None:
        betas = [float(r["beta"]) for r in rows if r.get("beta")]
        max_beta = max(betas) if betas else 0.0
        _check_above(max_beta, cfg.beta_end * 0.95,
                     "beta schedule reaches beta_end during training", fmt=".3f")

    # Best checkpoint epoch
    best_row = min(
        rows, key=lambda r: float(r.get("val_ckpt") or float("inf"))
    )
    _result(INFO,
            f"best checkpoint at epoch {best_row['epoch']}",
            f"val_ckpt={float(best_row['val_ckpt']):.4f}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-run validation (README §8)")
    p.add_argument("--save-dir", default="./saved_crl",
                   help="Directory containing checkpoints and crl_metrics.csv")
    p.add_argument("--checkpoint", default="crl_best.pth",
                   help="Checkpoint filename within --save-dir (default: crl_best.pth)")
    p.add_argument("--data-dir",    default="../data_files/parsed/train")
    p.add_argument("--val-dir",     default="../data_files/parsed/val")
    p.add_argument("--sensors",     nargs="+", default=None, choices=MODALITIES)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",      default=None, help="Force device: cpu / cuda / mps")
    p.add_argument("--out",         default=None,
                   help="JSON report path (default: <save-dir>/validation_report.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: GPU — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU (no GPU found)")

    # Sensors: prefer meta.json saved during training
    meta_path = save_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sensors = meta.get("sensors", MODALITIES)
        print(f"Loaded training meta: sensors={sensors}")
    else:
        sensors = args.sensors or MODALITIES
        print(f"No meta.json; using sensors={sensors}")

    cfg             = CRLConfig()
    cfg.batch_size  = args.batch_size
    cfg.num_workers = args.num_workers

    # Load checkpoint
    ckpt_path = save_dir / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    model = CRLModel(cfg, sensors=sensors).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Data loaders
    try:
        train_ds = SensorDataset(args.data_dir, cfg, is_train=True)
        val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False)
    except Exception as e:
        print(f"ERROR loading datasets: {e}", file=sys.stderr)
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_single,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_single,
    )

    sample_batch = next(iter(val_loader))
    primary      = "seismic" if "seismic" in sensors else sensors[0]

    # ----------------------------------------------------------------
    check_unit_level(model, sample_batch, device, cfg)

    # Collect latents once — reused by §8.2 and passed as arrays
    print("\n  Collecting latents over train and val splits...")
    z_tr,  vt_tr,  det_tr,  iv_tr  = _collect_latents(model, train_loader, device, primary)
    z_val, vt_val, det_val, iv_val = _collect_latents(model, val_loader,   device, primary)
    print(f"  Collected: train={len(z_tr)}, val={len(z_val)} samples")

    # ----------------------------------------------------------------
    eval_metrics = check_representation_quality(
        model,
        z_tr, vt_tr, det_tr, iv_tr,
        z_val, vt_val, det_val, iv_val,
        cfg, primary,
    )

    # ----------------------------------------------------------------
    check_causal_structure(model, sample_batch, device, cfg)

    # ----------------------------------------------------------------
    check_downstream_performance(
        model, val_loader, device, save_dir, primary, eval_metrics
    )

    # ----------------------------------------------------------------
    check_scientific_integrity(save_dir, cfg)

    # ----------------------------------------------------------------
    _header("VALIDATION SUMMARY")
    counts = {PASS: 0, FAIL: 0, WARN: 0, INFO: 0}
    for s, _, _ in _results:
        counts[s] += 1

    print(
        f"  {counts[PASS]} passed  |  "
        f"{counts[FAIL]} failed  |  "
        f"{counts[WARN]} warnings  |  "
        f"{counts[INFO]} info"
    )

    failed = [(n, d) for s, n, d in _results if s == FAIL]
    warned = [(n, d) for s, n, d in _results if s == WARN]

    if failed:
        print("\n  FAILURES:")
        for n, d in failed:
            print(f"    ✗ {n}" + (f" — {d}" if d else ""))
    if warned:
        print("\n  WARNINGS:")
        for n, d in warned:
            print(f"    ! {n}" + (f" — {d}" if d else ""))

    overall = "ALL CHECKS PASSED" if counts[FAIL] == 0 else f"{counts[FAIL]} CHECK(S) FAILED"
    print(f"\n  {overall}")

    # JSON report
    report_path = Path(args.out) if args.out else save_dir / "validation_report.json"
    report = {
        "checkpoint": str(ckpt_path),
        "summary": {k: int(v) for k, v in counts.items()},
        "results": [{"status": s, "name": n, "detail": d} for s, n, d in _results],
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    sys.exit(0 if counts[FAIL] == 0 else 1)


if __name__ == "__main__":
    main()
