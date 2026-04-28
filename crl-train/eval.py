"""CRL evaluation — runs the full pipeline on the test set and reports metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, CATEGORY_TO_IDX
from crl_vehicle.data.dataset import SensorDataset, collate_single
from crl_vehicle.probe.recalibration import (
    apply_binary_log_prior_shift,
    apply_multiclass_log_prior_shift,
    compute_binary_prior,
    compute_multiclass_prior,
)
from training.trainer import CRLModel

# Class index → display name
IDX_TO_CLASS = {v: k for k, v in CATEGORY_TO_IDX.items()}
N_TYPE_CLASSES = len(CATEGORY_TO_IDX)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CRL model on test set")
    p.add_argument("--save-dir",   required=True,
                   help="Run directory containing meta.json and downstream_best.pth")
    p.add_argument("--test-dir",   default="../data_files/parsed/test/")
    p.add_argument("--cache-dir",  default="./saved_crl/cache")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir",    default=None,
                   help="Where to write outputs (defaults to --save-dir)")
    p.add_argument("--include-datasets", nargs="+", default=None,
                   help="Only evaluate on these dataset prefixes (e.g. 'focal', 'iobt'). "
                        "Matches the 'dataset' field parsed from parquet stems. "
                        "When set, eval_report.json and confusion plots are written to "
                        "a subdirectory named by the filter.")
    p.add_argument("--recalibrate", action="store_true",
                   help="Also compute target-prior-calibrated metrics using the "
                        "ground-truth class distribution of this split as the target prior. "
                        "Adds 'presence_target_calibrated' and 'type_target_calibrated' "
                        "keys to eval_report.json. Assumes training used class-balanced "
                        "loss (uniform effective train prior). This is a diagnostic metric: "
                        "deployment does not know target priors, so do not quote these as "
                        "deployment numbers.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = (logits > 0).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-8)
    acc       = (tp + tn) / max(len(labels), 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    # Matthews correlation coefficient — 0 for degenerate predictors regardless
    # of class skew. Denominator underflow → 0.
    import math as _math
    mcc_den = _math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else 0.0
    return {
        "accuracy":  round(acc,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "specificity": round(specificity, 4),
        "f1":        round(f1,        4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "mcc":       round(mcc,       4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def multiclass_metrics(
    logits: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> dict:
    preds = logits.argmax(dim=-1)
    acc   = (preds == labels).float().mean().item()

    per_class: dict[str, dict] = {}
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = (2 * precision * recall) / max(precision + recall, 1e-8)
        per_class[IDX_TO_CLASS.get(c, str(c))] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int((labels == c).sum().item()),
        }

    macro_f1  = sum(v["f1"]        for v in per_class.values()) / n_classes
    macro_pre = sum(v["precision"] for v in per_class.values()) / n_classes
    macro_rec = sum(v["recall"]    for v in per_class.values()) / n_classes

    # support_only: average only over classes present in this split.
    # Filtered splits (focal, iobt) exclude some classes entirely; dividing by
    # n_classes in the unfiltered macro halves the apparent F1 for no model reason.
    present = [v for v in per_class.values() if v["support"] > 0]
    macro_f1_support_only = (
        sum(v["f1"] for v in present) / len(present) if present else 0.0
    )

    # Confusion matrix: rows = true, cols = predicted
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t][p] += 1

    return {
        "accuracy":              round(acc,                   4),
        "macro_f1":              round(macro_f1,              4),
        "macro_f1_support_only": round(macro_f1_support_only, 4),
        "macro_precision":       round(macro_pre,             4),
        "macro_recall":          round(macro_rec,             4),
        "per_class":             per_class,
        "confusion_matrix":      cm,
    }


def recalibrated_binary_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict:
    """binary_metrics computed after a log-prior shift to the split's empirical prior.

    Assumes class-balanced training (uniform effective train prior, p_train=0.5).
    """
    p_split = compute_binary_prior(labels)
    shifted = apply_binary_log_prior_shift(logits, p_split=p_split, p_train=0.5)
    m = binary_metrics(shifted, labels)
    m["p_split"] = round(p_split, 6)
    m["p_train_assumed"] = 0.5
    return m


def recalibrated_multiclass_metrics(
    logits: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> dict:
    """multiclass_metrics computed after a log-prior shift to the split's empirical prior.

    Assumes class-balanced training (uniform effective train prior,
    p_train=[1/K]*K). Classes with zero support in the split get an eps-floored
    prior to avoid log(0); effectively they are suppressed in the argmax.
    """
    p_split = compute_multiclass_prior(labels, n_classes=n_classes)
    shifted = apply_multiclass_log_prior_shift(logits, p_split=p_split)
    m = multiclass_metrics(shifted, labels, n_classes)
    m["p_split"] = [round(v, 6) for v in p_split.tolist()]
    m["p_train_assumed"] = [round(1.0 / n_classes, 6)] * n_classes
    return m


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = cm_arr / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 1.2 + 1),
                                    max(4, len(class_names) * 1.2)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = int(cm_arr[i, j])
            pct   = cm_norm[i, j]
            color = "white" if pct > 0.6 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_binary_confusion(
    tn: int, fp: int, fn: int, tp: int,
    title: str,
    out_path: Path,
) -> None:
    _plot_confusion_matrix(
        cm=[[tn, fp], [fn, tp]],
        class_names=["absent", "present"],
        title=title,
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: CRLModel,
    loader: DataLoader,
    device: torch.device,
    cfg: CRLConfig,
) -> dict[str, torch.Tensor]:
    model.eval()
    pres_logits: list[torch.Tensor] = []
    pres_labels: list[torch.Tensor] = []
    type_logits: list[torch.Tensor] = []
    type_labels: list[torch.Tensor] = []
    probe_mode = getattr(model, "probe_mode", "linear_ztype")
    use_fullz  = probe_mode == "linear_fullz"
    use_signal = probe_mode in ("linear_signal", "mlp_signal")
    d_signal   = model.cfg.d_signal

    def _select_type_slice(z_full, z_type_block, mask):
        if use_fullz:
            return z_full[mask]
        if use_signal:
            return z_full[mask][..., :d_signal]
        return z_type_block[mask]

    with torch.no_grad():
        for batch in loader:
            if model.is_fused_frontend():
                avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
                if not avail.any():
                    continue
                x_a = batch["x_audio"][avail].to(device)
                x_s = batch["x_seismic"][avail].to(device)
                _, z, _, _ = model.encode_fused(x_a, x_s)
                z_pres, z_type, _, _, _ = model.latent.split(z)
                det   = batch["detection_label"][avail].float()
                vtype = batch["vehicle_type"][avail]
                pres_logits.append(model.pres_heads["fused"](z_pres).squeeze(-1).cpu())
                pres_labels.append(det.long())
                valid = vtype >= 0
                if valid.any():
                    z_for_type = _select_type_slice(z, z_type, valid)
                    type_logits.append(model.type_heads["fused"](z_for_type).cpu())
                    type_labels.append(vtype[valid])
            else:
                for sensor in model.sensors:
                    avail = batch[f"{sensor}_avail"].bool()
                    if not avail.any():
                        continue
                    x = batch[f"x_{sensor}"][avail].to(device)
                    _, z, _, _ = model.encode(sensor, x)
                    z_pres, z_type, _, _, _ = model.latent.split(z)
                    det   = batch["detection_label"][avail].float()
                    vtype = batch["vehicle_type"][avail]
                    pres_logits.append(model.pres_heads[sensor](z_pres).squeeze(-1).cpu())
                    pres_labels.append(det.long())
                    valid = vtype >= 0
                    if valid.any():
                        z_for_type = _select_type_slice(z, z_type, valid)
                        type_logits.append(model.type_heads[sensor](z_for_type).cpu())
                        type_labels.append(vtype[valid])

    return {
        "pres_logits": torch.cat(pres_logits) if pres_logits else torch.empty(0),
        "pres_labels": torch.cat(pres_labels) if pres_labels else torch.empty(0, dtype=torch.long),
        "type_logits": torch.cat(type_logits) if type_logits else None,
        "type_labels": torch.cat(type_labels) if type_labels else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    save_dir = Path(args.save_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config from saved meta.json
    meta_path = save_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {save_dir}")
    meta = json.loads(meta_path.read_text())
    cfg_dict = meta.get("config", {})
    sensors  = meta.get("sensors", ["audio", "seismic"])
    probe_mode = meta.get("probe_mode", "linear_ztype")
    cfg = CRLConfig(**{k: v for k, v in cfg_dict.items() if hasattr(CRLConfig, k)
                       or k in CRLConfig.__dataclass_fields__})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load model
    ckpt_path = save_dir / "downstream_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"downstream_best.pth not found in {save_dir}. "
            "Run train.py --phase full (or --phase downstream) first."
        )
    model = CRLModel(cfg, sensors=sensors, probe_mode=probe_mode).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path} (probe_mode={probe_mode})")

    # Load test dataset
    cache_dir = Path(args.cache_dir)
    print(f"  Loading test set from {args.test_dir} …")
    test_ds = SensorDataset(args.test_dir, cfg, is_train=False, cache_dir=cache_dir)
    print(f"  {len(test_ds):,} test windows (pre-filter)")

    # Optional dataset filter — prune _index to only keep windows from matching datasets.
    # gkey is (ds, vehicle, rs, seg_key); index rows are (gkey, w, vtype, det, ...).
    if args.include_datasets:
        allowed = set(args.include_datasets)
        filtered = [row for row in test_ds._index if row[0][0] in allowed]
        dropped = len(test_ds._index) - len(filtered)
        test_ds._index = filtered
        print(f"  Dataset filter {sorted(allowed)}: kept {len(filtered):,} windows "
              f"({dropped:,} dropped)")
        if not filtered:
            raise RuntimeError(f"No windows remain after filter {sorted(allowed)}")
        # Redirect outputs to a subdirectory so multiple filtered runs don't collide.
        out_dir = out_dir / ("filter_" + "_".join(sorted(allowed)))
        out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_single,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # Run inference
    print("  Running inference …")
    outputs = run_inference(model, loader, device, cfg)

    # Compute metrics
    pres_m = binary_metrics(outputs["pres_logits"], outputs["pres_labels"])
    type_m = (
        multiclass_metrics(outputs["type_logits"], outputs["type_labels"], N_TYPE_CLASSES)
        if outputs["type_logits"] is not None and outputs["type_logits"].numel() > 0
        else None
    )

    pres_cal_m = None
    type_cal_m = None
    if args.recalibrate:
        pres_cal_m = recalibrated_binary_metrics(
            outputs["pres_logits"], outputs["pres_labels"]
        )
        if outputs["type_logits"] is not None and outputs["type_logits"].numel() > 0:
            type_cal_m = recalibrated_multiclass_metrics(
                outputs["type_logits"], outputs["type_labels"], N_TYPE_CLASSES
            )

    # Print summary
    print(f"\n{'=' * 55}")
    print("  PRESENCE DETECTION")
    print(f"{'=' * 55}")
    for k, v in pres_m.items():
        if k not in ("tp", "fp", "fn", "tn"):
            print(f"  {k:<18} {v}")

    if type_m:
        print(f"\n{'=' * 55}")
        print("  VEHICLE TYPE CLASSIFICATION")
        print(f"{'=' * 55}")
        print(f"  {'accuracy':<22} {type_m['accuracy']}")
        print(f"  {'macro_f1':<22} {type_m['macro_f1']}")
        print(f"  {'macro_f1_support_only':<22} {type_m['macro_f1_support_only']}")
        print(f"  {'macro_precision':<22} {type_m['macro_precision']}")
        print(f"  {'macro_recall':<22} {type_m['macro_recall']}")
        print(f"\n  Per-class:")
        for cls, vals in type_m["per_class"].items():
            print(f"    {cls:<14} f1={vals['f1']:.3f}  "
                  f"prec={vals['precision']:.3f}  rec={vals['recall']:.3f}  "
                  f"n={vals['support']}")

    if args.recalibrate and (pres_cal_m or type_cal_m):
        print(f"\n{'=' * 55}")
        print("  TARGET-CALIBRATED (log-prior shift, oracle split prior)")
        print("  Diagnostic only — deployment does not know target priors")
        print(f"{'=' * 55}")
        if pres_cal_m:
            # F1 is prior-sensitive and can be inflated by a "always-positive"
            # degenerate classifier under heavy skew. balanced_accuracy and MCC
            # are robust to this; report all three so the degeneracy is visible.
            print(f"  presence f1          = {pres_cal_m['f1']}")
            print(f"  presence bal_acc     = {pres_cal_m['balanced_accuracy']}")
            print(f"  presence mcc         = {pres_cal_m['mcc']}")
            print(f"           (p_split={pres_cal_m['p_split']})")
        if type_cal_m:
            print(f"  type     macro_f1    = {type_cal_m['macro_f1']} "
                  f"(support_only={type_cal_m['macro_f1_support_only']}, "
                  f"p_split={type_cal_m['p_split']})")

    # Save JSON report
    report = {
        "save_dir":  str(save_dir),
        "test_dir":  args.test_dir,
        "include_datasets": sorted(args.include_datasets) if args.include_datasets else None,
        "n_windows": len(test_ds),
        "presence":  pres_m,
        "type":      type_m,
    }
    if args.recalibrate:
        report["presence_target_calibrated"] = pres_cal_m
        report["type_target_calibrated"]     = type_cal_m
    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")

    # Confusion matrix plots
    pres_plot = out_dir / "confusion_presence.png"
    _plot_binary_confusion(
        tn=pres_m["tn"], fp=pres_m["fp"], fn=pres_m["fn"], tp=pres_m["tp"],
        title="Presence Detection (test set)",
        out_path=pres_plot,
    )
    print(f"  Plot saved:   {pres_plot}")

    if type_m:
        class_names = [IDX_TO_CLASS[i] for i in range(N_TYPE_CLASSES)]
        type_plot = out_dir / "confusion_type.png"
        _plot_confusion_matrix(
            cm=type_m["confusion_matrix"],
            class_names=class_names,
            title="Vehicle Type Classification (test set)",
            out_path=type_plot,
        )
        print(f"  Plot saved:   {type_plot}")


if __name__ == "__main__":
    main()
