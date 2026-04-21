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
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir",    default=None,
                   help="Where to write outputs (defaults to --save-dir)")
    p.add_argument("--include-datasets", nargs="+", default=None,
                   help="Only evaluate on these dataset prefixes (e.g. 'focal', 'iobt'). "
                        "Matches the 'dataset' field parsed from parquet stems. "
                        "When set, eval_report.json and confusion plots are written to "
                        "a subdirectory named by the filter.")
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
    f1        = (2 * precision * recall) / max(precision + recall, 1e-8)
    acc       = (tp + tn) / max(len(labels), 1)
    return {
        "accuracy":  round(acc,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
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

    # Confusion matrix: rows = true, cols = predicted
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t][p] += 1

    return {
        "accuracy":        round(acc,       4),
        "macro_f1":        round(macro_f1,  4),
        "macro_precision": round(macro_pre, 4),
        "macro_recall":    round(macro_rec, 4),
        "per_class":       per_class,
        "confusion_matrix": cm,
    }


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
    use_fullz = getattr(model, "probe_mode", "linear_ztype") == "linear_fullz"

    with torch.no_grad():
        for batch in loader:
            if cfg.frontend_type == "multiscale":
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
                    z_for_type = z[valid] if use_fullz else z_type[valid]
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
                        z_for_type = z[valid] if use_fullz else z_type[valid]
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
        print(f"  {'accuracy':<18} {type_m['accuracy']}")
        print(f"  {'macro_f1':<18} {type_m['macro_f1']}")
        print(f"  {'macro_precision':<18} {type_m['macro_precision']}")
        print(f"  {'macro_recall':<18} {type_m['macro_recall']}")
        print(f"\n  Per-class:")
        for cls, vals in type_m["per_class"].items():
            print(f"    {cls:<14} f1={vals['f1']:.3f}  "
                  f"prec={vals['precision']:.3f}  rec={vals['recall']:.3f}  "
                  f"n={vals['support']}")

    # Save JSON report
    report = {
        "save_dir":  str(save_dir),
        "test_dir":  args.test_dir,
        "include_datasets": sorted(args.include_datasets) if args.include_datasets else None,
        "n_windows": len(test_ds),
        "presence":  pres_m,
        "type":      type_m,
    }
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
