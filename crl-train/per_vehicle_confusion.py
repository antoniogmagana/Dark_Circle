#!/usr/bin/env python3
"""Per-vehicle confusion analysis on a trained downstream probe.

Loads a downstream-trained model from `<run_dir>/downstream/<probe>__<ckpt>/`
plus its CRL checkpoint, runs inference on the test set, and groups
predictions by the (dataset, vehicle, rs_node) of each window. Produces:

  per_vehicle_confusion.csv
    one row per (dataset, vehicle), with:
      true_class, n_windows, n_rs_nodes,
      pred_pedestrian, pred_light, pred_medium, pred_heavy   (fractions)
      pred_majority, majority_frac, accuracy
      reclass_recommendation  (only if pred_majority != true_class
                                AND majority_frac >= --reclass-threshold)

  per_vehicle_per_rs.csv
    same but split out per (dataset, vehicle, rs_node)

  per_vehicle_confusion.json
    the same data plus model/run metadata for reproducibility

Usage
-----
    # Default: ID-split test set, the strongest run we have
    python per_vehicle_confusion.py \
        --downstream-dir saved_crl/runs/multiscale/vae/v1_diag/downstream/linear_fullz__crl_best_aux_type \
        --use-id-split --id-root ../data_files/parsed/

Caveats
-------
- Fused frontends only (multiscale, morlet_fused). Per-sensor frontends emit
  two logit rows per sample which breaks the index→prediction mapping; the
  script will refuse to run on those and tell you why.
- Requires shuffle=False on the loader (default for eval) — relies on
  dataset._index ordering to recover per-window vehicle identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import (
    CATEGORY_TO_IDX,
    DATASET_VEHICLE_MAP,
    CRLConfig,
)
from crl_vehicle.data.dataset import SensorDataset, collate_single
from crl_vehicle.seeding import seed_everything, seeded_dataloader_kwargs
from training.trainer import CRLModel

IDX_TO_CLASS = {v: k for k, v in CATEGORY_TO_IDX.items()}
N_CLASSES = len(CATEGORY_TO_IDX)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--downstream-dir",
        required=True,
        type=Path,
        help="Path to a downstream/<probe>__<ckpt>/ directory "
        "containing meta.json + downstream_best.pth.",
    )
    p.add_argument("--test-dir", default="../data_files/parsed/test/")
    p.add_argument("--cache-dir", default="./saved_crl/caches/waveform")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output dir (defaults to --downstream-dir/per_vehicle/)",
    )
    p.add_argument(
        "--use-id-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eval against the ID-split test partition (default). "
        "Pass --no-use-id-split to use --test-dir instead.",
    )
    p.add_argument("--id-root", default="../data_files/parsed/")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--reclass-threshold",
        type=float,
        default=0.50,
        help="Flag a vehicle for reclassification when its predicted "
        "majority class differs from its labeled class AND the "
        "majority class wins ≥ this fraction of windows. "
        "0.50 is conservative; lower for more candidates.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (default 42). Persisted into the "
        "per_vehicle_confusion.json output.",
    )
    return p.parse_args()


def load_model_and_cfg(
    downstream_dir: Path, device: torch.device
) -> tuple[CRLModel, CRLConfig, list[str]]:
    """Load the downstream-trained model from a downstream/<probe>__<ckpt>/ dir."""
    meta = json.loads((downstream_dir / "meta.json").read_text())
    cfg = CRLConfig(
        **{k: v for k, v in meta["config"].items() if k in CRLConfig.__dataclass_fields__}
    )
    sensors = meta["sensors"]
    probe_mode = meta.get("probe_mode", "linear_ztype")
    model = CRLModel(cfg, sensors=sensors, probe_mode=probe_mode).to(device)
    state = torch.load(downstream_dir / "downstream_best.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    if not model.is_fused_frontend():
        raise SystemExit(
            f"This script only supports fused frontends (multiscale, morlet_fused). "
            f"Got frontend_type={cfg.frontend_type!r}, which is per-sensor — "
            f"run_inference emits two rows per sample for those, breaking the "
            f"per-window index mapping this script depends on."
        )
    return model, cfg, sensors


def make_test_dataset(args, cfg: CRLConfig) -> SensorDataset:
    cache_dir = Path(args.cache_dir)
    if args.use_id_split:
        return SensorDataset(
            args.test_dir,
            cfg,
            is_train=False,
            cache_dir=cache_dir,
            use_id_split=True,
            role="test",
            id_root=args.id_root,
            id_cache_dir=Path("saved_crl/caches/id_split"),
        )
    return SensorDataset(args.test_dir, cfg, is_train=False, cache_dir=cache_dir)


@torch.no_grad()
def run_per_window_inference(
    model: CRLModel,
    ds: SensorDataset,
    dev: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[list[tuple[str, str, str]], torch.Tensor, torch.Tensor]:
    """Returns (per_window_keys, type_logits, type_labels), aligned by index.

    per_window_keys[i] = (dataset, vehicle, rs_node) for window i in dataset order.
    Only windows with valid type label (vehicle_type >= 0) are emitted.
    """
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_single,
        pin_memory=True,
        **seeded_dataloader_kwargs(seed),
    )
    model.eval()
    probe_mode = getattr(model, "probe_mode", "linear_ztype")
    use_fullz = probe_mode == "linear_fullz"
    use_signal = probe_mode in ("linear_signal", "mlp_signal")
    d_signal = model.cfg.d_signal

    def select_z(z_full, z_type_block, mask):
        if use_fullz:
            return z_full[mask]
        if use_signal:
            return z_full[mask][..., :d_signal]
        return z_type_block[mask]

    all_logits, all_labels, all_global_idx = [], [], []
    cursor = 0
    for batch in loader:
        bsz = batch["vehicle_type"].numel()
        avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
        # Local indices into the batch for available samples
        avail_local = avail.nonzero(as_tuple=True)[0]
        # Map back to dataset indices
        avail_global = (cursor + avail_local).tolist()
        if avail.any():
            x_a = batch["x_audio"][avail].to(dev)
            x_s = batch["x_seismic"][avail].to(dev)
            _, z, _, _ = model.encode_fused(x_a, x_s)
            z_pres, z_type, _, _, _ = model.latent.split(z)
            vtype = batch["vehicle_type"][avail]
            valid = vtype >= 0
            if valid.any():
                z_for_type = select_z(z, z_type, valid)
                logits = model.type_heads["fused"](z_for_type).cpu()
                all_logits.append(logits)
                all_labels.append(vtype[valid])
                # Recover global indices for the valid-typed available samples
                valid_local_in_avail = valid.nonzero(as_tuple=True)[0].tolist()
                all_global_idx.extend(avail_global[i] for i in valid_local_in_avail)
        cursor += bsz

    if not all_logits:
        raise RuntimeError("No valid-typed windows produced any logits — empty test set?")

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Recover (dataset, vehicle, rs_node) per global index from dataset._index
    keys: list[tuple[str, str, str]] = []
    for gi in all_global_idx:
        gkey, _w, _vt, _det, _aid, _sid = ds._index[gi]
        ds_name, vehicle, rs, _seg = gkey
        keys.append((ds_name, vehicle, rs))

    assert len(keys) == logits.shape[0] == labels.shape[0]
    return keys, logits, labels


def aggregate(keys, preds, labels, group_by="vehicle"):
    """group_by ∈ {'vehicle', 'vehicle_rs'}."""
    buckets = defaultdict(
        lambda: {
            "true_class": None,
            "n": 0,
            "rs_nodes": set(),
            "pred_counts": [0] * N_CLASSES,
            "n_correct": 0,
        }
    )
    for (ds, v, rs), pred, label in zip(keys, preds.tolist(), labels.tolist(), strict=False):
        if group_by == "vehicle":
            key = (ds, v)
        else:
            key = (ds, v, rs)
        b = buckets[key]
        b["true_class"] = int(label)  # all windows of one vehicle share the class
        b["n"] += 1
        b["rs_nodes"].add(rs)
        b["pred_counts"][int(pred)] += 1
        if int(pred) == int(label):
            b["n_correct"] += 1
    return buckets


def write_csv(out_path: Path, buckets: dict, group_by: str, reclass_thresh: float) -> list[dict]:
    rows = []
    for key, b in sorted(buckets.items()):
        if group_by == "vehicle":
            ds, v = key
            rs = ""
        else:
            ds, v, rs = key
        n = b["n"]
        true_idx = b["true_class"]
        true_name = IDX_TO_CLASS[true_idx]
        pred_fracs = [c / n for c in b["pred_counts"]]
        pred_majority = int(max(range(N_CLASSES), key=lambda c: b["pred_counts"][c]))
        majority_frac = pred_fracs[pred_majority]
        acc = b["n_correct"] / n
        recommendation = ""
        if pred_majority != true_idx and majority_frac >= reclass_thresh:
            recommendation = (
                f"consider reclassifying as {IDX_TO_CLASS[pred_majority]} "
                f"({majority_frac:.0%} of windows predicted that)"
            )
        row = {
            "dataset": ds,
            "vehicle": v,
            "rs_node": rs,
            "true_class": true_name,
            "n_windows": n,
            "n_rs_nodes": len(b["rs_nodes"]),
            "pred_pedestrian": round(pred_fracs[0], 4),
            "pred_light": round(pred_fracs[1], 4),
            "pred_medium": round(pred_fracs[2], 4),
            "pred_heavy": round(pred_fracs[3], 4),
            "pred_majority": IDX_TO_CLASS[pred_majority],
            "majority_frac": round(majority_frac, 4),
            "accuracy": round(acc, 4),
            "reclass_recommendation": recommendation,
        }
        rows.append(row)

    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return rows


def print_summary(rows_by_vehicle: list[dict], reclass_thresh: float) -> None:
    print()
    print(
        f"{'dataset':<7} {'vehicle':<25} {'true':<11} "
        f"{'ped':>5} {'lit':>5} {'med':>5} {'hvy':>5} "
        f"{'pred':<11} {'maj%':>5} {'acc':>5}  recommendation"
    )
    print("-" * 110)
    flagged = []
    for r in rows_by_vehicle:
        line = (
            f"{r['dataset']:<7} {r['vehicle']:<25} {r['true_class']:<11} "
            f"{r['pred_pedestrian']:>5.2f} {r['pred_light']:>5.2f} "
            f"{r['pred_medium']:>5.2f} {r['pred_heavy']:>5.2f} "
            f"{r['pred_majority']:<11} {r['majority_frac']:>5.0%} "
            f"{r['accuracy']:>5.0%}  "
            f"{r['reclass_recommendation']}"
        )
        print(line)
        if r["reclass_recommendation"]:
            flagged.append(r)

    print()
    print(
        f"=== {len(flagged)} vehicles flagged for reclassification "
        f"(majority_frac ≥ {reclass_thresh:.0%}) ==="
    )
    for r in flagged:
        print(
            f"  {r['dataset']}/{r['vehicle']}: "
            f"{r['true_class']} → {r['pred_majority']} "
            f"({r['majority_frac']:.0%} of {r['n_windows']} windows)"
        )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else (args.downstream_dir / "per_vehicle")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.downstream_dir} …")
    model, cfg, sensors = load_model_and_cfg(args.downstream_dir, dev)
    print(
        f"  frontend={cfg.frontend_type}, probe_mode={getattr(model, 'probe_mode', '?')}, "
        f"d_z={cfg.d_z}, d_signal={cfg.d_signal}"
    )

    print(f"\nLoading test dataset (use_id_split={args.use_id_split}) …")
    test_ds = make_test_dataset(args, cfg)
    print(f"  {len(test_ds):,} test windows")

    print("\nRunning per-window inference …")
    keys, logits, labels = run_per_window_inference(
        model,
        test_ds,
        dev,
        args.batch_size,
        args.num_workers,
        seed=args.seed,
    )
    preds = logits.argmax(dim=-1)
    overall_acc = (preds == labels).float().mean().item()
    print(f"  {len(keys):,} valid-typed windows  |  overall acc = {overall_acc:.4f}")

    print("\nAggregating by vehicle …")
    by_vehicle = aggregate(keys, preds, labels, group_by="vehicle")
    by_rs = aggregate(keys, preds, labels, group_by="vehicle_rs")

    # Coverage report: which vehicles in DATASET_VEHICLE_MAP produced zero
    # test windows? Tells you up front when the test partition is missing
    # whole datasets (e.g. m3nvc under id-split + split_runs partitioner).
    seen_vehicles = set(by_vehicle.keys())
    expected_vehicles = {
        (ds, v)
        for ds, vmap in DATASET_VEHICLE_MAP.items()
        for v, entry in vmap.items()
        if v != "background" and not (ds == "m3nvc" and "_" in v)
    }
    missing = sorted(expected_vehicles - seen_vehicles)
    if missing:
        print(f"\n*** {len(missing)} vehicles in DATASET_VEHICLE_MAP have ZERO test windows: ***")
        by_ds = defaultdict(list)
        for ds, v in missing:
            by_ds[ds].append(v)
        for ds in sorted(by_ds):
            print(f"  {ds}: {', '.join(by_ds[ds])}")
        print("    (under id-split, missing vehicles are routed to other roles by")
        print("     plain markers or by the split_runs partitioner)")

    rows_v = write_csv(
        out_dir / "per_vehicle_confusion.csv",
        by_vehicle,
        "vehicle",
        args.reclass_threshold,
    )
    rows_rs = write_csv(
        out_dir / "per_vehicle_per_rs.csv", by_rs, "vehicle_rs", args.reclass_threshold
    )

    summary = {
        "downstream_dir": str(args.downstream_dir),
        "n_test_windows": len(test_ds),
        "n_typed_windows": len(keys),
        "overall_accuracy": round(overall_acc, 4),
        "reclass_threshold": args.reclass_threshold,
        "frontend_type": cfg.frontend_type,
        "probe_mode": getattr(model, "probe_mode", None),
        "use_id_split": args.use_id_split,
        "seed": args.seed,
        "per_vehicle": rows_v,
        "per_vehicle_per_rs": rows_rs,
    }
    (out_dir / "per_vehicle_confusion.json").write_text(json.dumps(summary, indent=2))

    print_summary(rows_v, args.reclass_threshold)
    print(f"\nWrote {out_dir / 'per_vehicle_confusion.csv'}")
    print(f"Wrote {out_dir / 'per_vehicle_per_rs.csv'}")
    print(f"Wrote {out_dir / 'per_vehicle_confusion.json'}")


if __name__ == "__main__":
    main()
