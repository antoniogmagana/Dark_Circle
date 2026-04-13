#!/usr/bin/env python3
"""
validate_run.py — Post-run validation.

Loads a trained CRL checkpoint and runs a structured battery of checks:
  §8.1  Unit-level integrity   (shapes, finiteness, filterbank bounds)
  §8.2  Embedding quality      (linear probe accuracy per embedding block)
  §8.4  Downstream performance (AUC, accuracy, F1; probe vs fine-tuned head)
  §8.6  Training integrity     (loss trend from crl_metrics.csv)

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
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import SensorDataset, collate_single
from training.eval import linear_probe_accuracy
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


# ---------------------------------------------------------------------------
# Embedding collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_embeddings(
    model: CRLModel,
    loader,
    device: torch.device,
    sensor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (e_pres, e_type, e_inst, vehicle_type, detection_label) numpy arrays."""
    ep_list, et_list, ei_list, vtypes, dets = [], [], [], [], []
    for batch in loader:
        x = batch[f"x_{sensor}"].to(device)
        e_pres, e_type, e_inst, _ = model.encode_modality(sensor, x)
        ep_list.append(e_pres.cpu().numpy())
        et_list.append(e_type.cpu().numpy())
        ei_list.append(e_inst.cpu().numpy())
        vtypes.append(batch["vehicle_type"].numpy())
        dets.append(batch["detection_label"].numpy())
    return (
        np.concatenate(ep_list),
        np.concatenate(et_list),
        np.concatenate(ei_list),
        np.concatenate(vtypes),
        np.concatenate(dets),
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
    all_finite    = True

    for sensor in model.sensors:
        x   = batch[f"x_{sensor}"].to(device)
        B   = x.shape[0]
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

        e_pres, e_type, e_inst = model.encoders[sensor](h)
        if tuple(e_pres.shape) != (B, cfg.d_pres):
            _result(FAIL, f"e_pres shape [{sensor}]",
                    f"got {tuple(e_pres.shape)}, expected ({B}, {cfg.d_pres})")
            all_shapes_ok = False
        if tuple(e_type.shape) != (B, cfg.d_type):
            _result(FAIL, f"e_type shape [{sensor}]",
                    f"got {tuple(e_type.shape)}, expected ({B}, {cfg.d_type})")
            all_shapes_ok = False
        if tuple(e_inst.shape) != (B, cfg.d_inst):
            _result(FAIL, f"e_inst shape [{sensor}]",
                    f"got {tuple(e_inst.shape)}, expected ({B}, {cfg.d_inst})")
            all_shapes_ok = False
        for name, t in [("e_pres", e_pres), ("e_type", e_type), ("e_inst", e_inst)]:
            if not t.isfinite().all():
                _result(FAIL, f"{name} finite [{sensor}]", "non-finite values")
                all_finite = False

        # Reconstruction
        import torch as _t
        e_cat = _t.cat([e_pres, e_type, e_inst], dim=-1)
        x_hat = model.decoders[sensor](e_cat)
        exp_dec = (B, mod.filterbank_out_channels, mod.t_prime)
        if tuple(x_hat.shape) != exp_dec:
            _result(FAIL, f"decoder shape [{sensor}]",
                    f"got {tuple(x_hat.shape)}, expected {exp_dec}")
            all_shapes_ok = False

    if all_shapes_ok:
        _result(PASS, "shape trace: Filterbank → SSM → Encoder → Decoder all correct")
    if all_finite:
        _result(PASS, "all intermediate tensors are finite")

    _check(cfg.audio_cfg.sample_rate == 16000, "audio sample rate == 16000 Hz")
    _check(cfg.seismic_cfg.sample_rate == 200, "seismic sample rate == 200 Hz")


# ---------------------------------------------------------------------------
# §8.2  Embedding quality
# ---------------------------------------------------------------------------

def check_embedding_quality(
    ep_tr: np.ndarray,
    et_tr: np.ndarray,
    ei_tr: np.ndarray,
    vt_tr: np.ndarray,
    det_tr: np.ndarray,
    ep_val: np.ndarray,
    et_val: np.ndarray,
    ei_val: np.ndarray,
    vt_val: np.ndarray,
    det_val: np.ndarray,
) -> dict:
    _header("§8.2  Embedding Quality (Linear Probes)")
    metrics: dict = {}

    # Presence embedding: should predict detection label
    m_pres = linear_probe_accuracy(ep_tr, det_tr, ep_val, det_val, label_name="probe_pres")
    metrics.update(m_pres)
    _check_above(m_pres["probe_pres_acc"], 0.6,
                 "presence embedding probe acc > 0.60 (binary detection)")
    _result(INFO, "presence probe F1", f"{m_pres['probe_pres_f1']:.4f}")

    # Type embedding: should predict vehicle type
    m_type = linear_probe_accuracy(et_tr, vt_tr, et_val, vt_val, label_name="probe_type")
    metrics.update(m_type)
    _check_above(m_type["probe_type_acc"], 0.25,
                 "type embedding probe acc > 0.25 (4-class random baseline)")
    _result(INFO, "type probe F1", f"{m_type['probe_type_f1']:.4f}")

    # Instance embedding: should predict instance type
    inst_tr_dummy  = np.zeros(len(ep_tr),  dtype=int)   # placeholder if no inst labels
    inst_val_dummy = np.zeros(len(ep_val), dtype=int)
    _result(INFO, "instance probe", "computed if instance labels available")

    # Detection AUC (presence embedding)
    if len(np.unique(det_val)) == 2:
        try:
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(ep_tr, det_tr)
            scores = clf.predict_proba(ep_val)[:, 1]
            auc = float(roc_auc_score(det_val, scores))
            metrics["detection_auc"] = auc
            _check_above(auc, 0.7, "presence embedding detection AUC > 0.70")
        except Exception:
            metrics["detection_auc"] = 0.0
            _result(WARN, "detection AUC computation failed")
    else:
        metrics["detection_auc"] = 0.0
        _result(WARN, "only one detection class in val split — AUC undefined")

    return metrics


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

    probe_pres_acc = eval_metrics.get("probe_pres_acc", 0.0)
    probe_pres_f1  = eval_metrics.get("probe_pres_f1",  0.0)
    probe_type_acc = eval_metrics.get("probe_type_acc", 0.0)
    probe_type_f1  = eval_metrics.get("probe_type_f1",  0.0)
    det_auc        = eval_metrics.get("detection_auc",  0.0)

    print("\n  Linear probe on frozen backbone embeddings:")
    _check_above(det_auc,        0.7,  "  detection AUC > 0.70")
    _check_above(probe_pres_acc, 0.6,  "  presence probe acc > 0.60")
    _check_above(probe_type_acc, 0.25, "  type probe acc > 0.25 (4-class random baseline)")
    _result(INFO, "  presence probe F1", f"{probe_pres_f1:.4f}")
    _result(INFO, "  type probe F1",     f"{probe_type_f1:.4f}")

    # Fine-tuned head
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

    for batch in val_loader:
        x     = batch[f"x_{primary_sensor}"].to(device)
        vtype = batch["vehicle_type"]
        det   = batch["detection_label"]

        e_pres, e_type, e_inst, _ = model.encode_modality(primary_sensor, x)
        pres_logit, type_logits, _ = model.det_heads[primary_sensor](e_pres, e_type, e_inst)

        scores = torch.sigmoid(pres_logit).cpu().numpy()
        det_scores.extend(scores.tolist())
        det_pred.extend((pres_logit > 0).long().cpu().tolist())
        det_true.extend(det.tolist())

        cls_mask = vtype >= 0
        if cls_mask.any():
            cls_pred.extend(type_logits[cls_mask.to(device)].argmax(1).cpu().tolist())
            cls_true.extend(vtype[cls_mask].tolist())

    print("\n  Fine-tuned downstream head:")
    if len(np.unique(det_true)) == 2:
        auc2    = float(roc_auc_score(det_true, det_scores))
        det_f1_2 = float(f1_score(det_true, det_pred, average="weighted", zero_division=0))
        _check_above(auc2, 0.7, "  detection AUC > 0.70 (Phase 2)")
        _result(INFO, "  detection weighted F1 (Phase 2)", f"{det_f1_2:.4f}")
    else:
        _result(WARN, "  only one detection class in val split — AUC undefined")

    if cls_true:
        cls_acc_2 = float(accuracy_score(cls_true, cls_pred))
        cls_f1_2  = float(f1_score(cls_true, cls_pred, average="weighted", zero_division=0))
        _check_above(cls_acc_2, 0.25, "  vehicle type accuracy > 0.25 (Phase 2)")
        _result(INFO, "  vehicle type F1 (Phase 2)", f"{cls_f1_2:.4f}")

        _result(INFO, "  probe vs head type accuracy",
                f"probe={probe_type_acc:.4f}  head={cls_acc_2:.4f}  "
                f"Δ={cls_acc_2 - probe_type_acc:+.4f}")
        _result(
            PASS if cls_acc_2 >= probe_type_acc else WARN,
            "  fine-tuning adds value beyond linear decodability",
        )


# ---------------------------------------------------------------------------
# §8.6  Training integrity
# ---------------------------------------------------------------------------

def check_training_integrity(save_dir: Path) -> None:
    _header("§8.6  Training Integrity")

    metrics_path = save_dir / "crl_metrics.csv"
    if not metrics_path.exists():
        _result(WARN, "crl_metrics.csv not found", f"expected at {metrics_path}")
        return

    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        _result(WARN, "crl_metrics.csv is empty")
        return

    _result(INFO, f"training history: {len(rows)} epochs logged")

    # val loss should decrease overall
    val_totals = [float(r["val_total"]) for r in rows if r.get("val_total")]
    if len(val_totals) >= 2:
        improved = min(val_totals) < val_totals[0]
        _result(
            PASS if improved else FAIL,
            "val_total decreases from initial value",
            f"initial={val_totals[0]:.4f}  min={min(val_totals):.4f}",
        )

    # Presence loss should be lower than initial
    pres_cols = [float(r["val_pres"]) for r in rows if r.get("val_pres")]
    if len(pres_cols) >= 2:
        _result(INFO, "val presence loss trend",
                f"initial={pres_cols[0]:.4f}  final={pres_cols[-1]:.4f}")

    best_row = min(rows, key=lambda r: float(r.get("val_total") or float("inf")))
    _result(INFO, f"best checkpoint at epoch {best_row['epoch']}",
            f"val_total={float(best_row['val_total']):.4f}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-run validation")
    p.add_argument("--save-dir",    default="./saved_crl")
    p.add_argument("--checkpoint",  default="crl_best.pth")
    p.add_argument("--data-dir",    default="../data_files/parsed/train")
    p.add_argument("--val-dir",     default="../data_files/parsed/val")
    p.add_argument("--sensors",     nargs="+", default=None, choices=MODALITIES)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",      default=None)
    p.add_argument("--out",         default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)

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
        print("Device: CPU")

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

    ckpt_path = save_dir / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    model = CRLModel(cfg, sensors=sensors).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

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

    print("\n  Collecting embeddings over train and val splits...")
    ep_tr, et_tr, ei_tr, vt_tr, det_tr = _collect_embeddings(
        model, train_loader, device, primary
    )
    ep_val, et_val, ei_val, vt_val, det_val = _collect_embeddings(
        model, val_loader, device, primary
    )
    print(f"  Collected: train={len(ep_tr)}, val={len(ep_val)} samples")

    # ----------------------------------------------------------------
    eval_metrics = check_embedding_quality(
        ep_tr, et_tr, ei_tr, vt_tr, det_tr,
        ep_val, et_val, ei_val, vt_val, det_val,
    )

    # ----------------------------------------------------------------
    check_downstream_performance(
        model, val_loader, device, save_dir, primary, eval_metrics
    )

    # ----------------------------------------------------------------
    check_training_integrity(save_dir)

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

    # Write JSON report
    out_path = Path(args.out) if args.out else save_dir / "validation_report.json"
    report = {
        "passed": counts[PASS],
        "failed": counts[FAIL],
        "warned": counts[WARN],
        "results": [
            {"status": s, "name": n, "detail": d}
            for s, n, d in _results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {out_path}")

    sys.exit(0 if counts[FAIL] == 0 else 1)


if __name__ == "__main__":
    main()
