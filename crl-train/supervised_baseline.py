#!/usr/bin/env python3
"""End-to-end supervised classifier — diagnostic baseline for CRL ceiling.

Reuses the CRL frontend + transformer encoder, but drops the entire VAE
machinery: classifies directly off the encoder mean (mu) with a 2-layer MLP,
trained end-to-end with class-weighted cross-entropy. No KL, no aux losses,
no contrastive, no probe phase.

Purpose
-------
Tells you what the data + frontend + encoder architecture can achieve for
type classification when nothing else gets in the way. Compare to CRL's
val_type_f1 / test type_macro_f1 to know whether CRL is leaving signal on
the table or whether the data itself caps you.

Usage
-----
    # ID-split, multiscale, 30 epochs:
    python supervised_baseline.py \
        --frontend multiscale \
        --use-id-split --id-root ../data_files/parsed/ \
        --epochs 30 --out-dir saved_crl/supervised_multiscale

    # File-split parity with run_full_diagnostic defaults:
    python supervised_baseline.py --frontend multiscale --epochs 30 \
        --out-dir saved_crl/supervised_filesplit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    SensorDataset, collate_single, compute_class_weights,
)
from training.trainer import CRLModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir",   default="../data_files/parsed/train/")
    p.add_argument("--val-dir",    default="../data_files/parsed/val/")
    p.add_argument("--test-dir",   default="../data_files/parsed/test/")
    p.add_argument("--cache-dir",  default="./saved_crl/cache")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--use-id-split", action="store_true")
    p.add_argument("--id-root",    default="../data_files/parsed/")
    p.add_argument("--frontend",
                   choices=["multiscale", "morlet", "morlet_per_sensor",
                            "morlet_fused"],
                   default="multiscale")
    p.add_argument("--morlet-use-phase", action="store_true")
    p.add_argument("--sensors", nargs="+", default=["audio", "seismic"])
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--num-workers", type=int,  default=4)
    p.add_argument("--head-hidden", type=int,  default=64,
                   help="MLP head hidden dim. Set 0 for a single Linear.")
    return p.parse_args()


class SupervisedClassifier(nn.Module):
    """Wraps a CRLModel encoder and replaces the VAE outputs with a type head.

    Reads mu from the encoder (deterministic, no sampling) and feeds it to
    a 2-layer MLP. Training is end-to-end across frontend + encoder + head.
    """
    def __init__(self, crl: CRLModel, d_z: int, head_hidden: int = 64,
                 n_classes: int = 4) -> None:
        super().__init__()
        self.crl = crl
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(d_z, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, n_classes),
            )
        else:
            self.head = nn.Linear(d_z, n_classes)

    def encode_mu(self, batch: dict, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, valid_mask) over the batch.

        For fused frontends, requires both sensors available per sample.
        For per-sensor frontends, averages mu across the available sensors
        per sample (mask reflects 'at least one sensor available').
        """
        if self.crl.is_fused_frontend():
            avail = batch["audio_avail"].bool() & batch["seismic_avail"].bool()
            if not avail.any():
                return (torch.empty(0, device=dev), avail)
            x_a = batch["x_audio"][avail].to(dev)
            x_s = batch["x_seismic"][avail].to(dev)
            _, _, mu, _ = self.crl.encode_fused(x_a, x_s)
            return mu, avail

        # Per-sensor: average mu across sensors a sample has.
        mus_per_sample: dict[int, list[torch.Tensor]] = {}
        for sensor in self.crl.sensors:
            avail = batch[f"{sensor}_avail"].bool()
            if not avail.any():
                continue
            x = batch[f"x_{sensor}"][avail].to(dev)
            _, _, mu_s, _ = self.crl.encode(sensor, x)
            for local_i, global_i in enumerate(avail.nonzero(as_tuple=True)[0].tolist()):
                mus_per_sample.setdefault(global_i, []).append(mu_s[local_i])

        if not mus_per_sample:
            return (torch.empty(0, device=dev),
                    torch.zeros(len(batch["vehicle_type"]), dtype=torch.bool))
        order = sorted(mus_per_sample.keys())
        mus = torch.stack([torch.stack(mus_per_sample[i]).mean(dim=0)
                           for i in order])
        mask = torch.zeros(len(batch["vehicle_type"]), dtype=torch.bool)
        mask[order] = True
        return mus, mask

    def forward(self, batch: dict, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mu, mask = self.encode_mu(batch, dev)
        if mu.numel() == 0:
            return mu, mask
        return self.head(mu), mask


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, n_classes: int = 4) -> tuple[float, float, list[float]]:
    """Returns (macro_f1, accuracy, per_class_f1)."""
    if logits.numel() == 0:
        return 0.0, 0.0, [0.0] * n_classes
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    per_class = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        per_class.append((2 * tp) / max(2 * tp + fp + fn, 1))
    return sum(per_class) / n_classes, acc, per_class


def make_dataset(args, cfg, role: str, parquet_dir: str, is_train: bool) -> SensorDataset:
    cache_dir = Path(args.cache_dir)
    if args.use_id_split:
        return SensorDataset(
            parquet_dir, cfg, is_train=is_train, cache_dir=cache_dir,
            use_id_split=True, role=role,
            id_root=args.id_root, id_cache_dir=Path("saved_crl/id_cache"),
        )
    return SensorDataset(parquet_dir, cfg, is_train=is_train, cache_dir=cache_dir)


def run_one_pass(model: SupervisedClassifier, loader: DataLoader,
                 dev: torch.device, opt: torch.optim.Optimizer | None,
                 type_weights: torch.Tensor | None) -> tuple[float, float, float, list[float]]:
    is_train = opt is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    all_logits, all_labels = [], []
    for batch in loader:
        if is_train:
            opt.zero_grad()
        logits, mask = model(batch, dev)
        if logits.numel() == 0:
            continue
        labels_full = batch["vehicle_type"].to(dev)
        labels = labels_full[mask.to(dev)] if mask.any() else labels_full
        valid = labels >= 0
        if not valid.any():
            continue
        logits, labels = logits[valid], labels[valid].long()
        loss = F.cross_entropy(logits, labels, weight=type_weights)
        if is_train:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    if not all_logits:
        return 0.0, 0.0, 0.0, [0.0] * 4
    f1, acc, per_class = macro_f1(torch.cat(all_logits), torch.cat(all_labels))
    return total_loss / max(n_batches, 1), f1, acc, per_class


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = CRLConfig(
        frontend_type=args.frontend,
        morlet_use_phase=args.morlet_use_phase,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        num_workers=args.num_workers,
        n_epochs=args.epochs,
    )
    if args.use_id_split:
        cfg.use_id_split = True

    print(f"\nPreloading datasets (use_id_split={args.use_id_split}) …")
    t0 = time.time()
    train_ds = make_dataset(args, cfg, "train", args.data_dir, is_train=True)
    val_ds   = make_dataset(args, cfg, "val",   args.val_dir,  is_train=False)
    test_ds  = make_dataset(args, cfg, "test",  args.test_dir, is_train=False)
    print(f"  Done in {(time.time()-t0)/60:.1f} min  "
          f"({len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test)")

    pres_w, type_w = compute_class_weights(train_ds)
    type_w = type_w.to(dev)
    print(f"  Class weights — type: {[round(w, 3) for w in type_w.tolist()]}")

    crl = CRLModel(cfg, sensors=args.sensors, probe_mode="linear_ztype").to(dev)
    model = SupervisedClassifier(crl, d_z=cfg.d_z, head_hidden=args.head_hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_single,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_single,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_single,
                              pin_memory=True)

    csv_path = out_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "train_f1", "train_acc",
              "val_loss", "val_f1", "val_acc",
              "ped_f1", "light_f1", "medium_f1", "heavy_f1"]
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fields)
    writer.writeheader()

    best_val_f1 = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        t_e = time.time()
        tr_loss, tr_f1, tr_acc, _ = run_one_pass(model, train_loader, dev, opt, type_w)
        with torch.no_grad():
            va_loss, va_f1, va_acc, va_per = run_one_pass(model, val_loader, dev, None, type_w)
        writer.writerow({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_f1": round(tr_f1, 4), "train_acc": round(tr_acc, 4),
            "val_loss":   round(va_loss, 4), "val_f1":   round(va_f1, 4), "val_acc":   round(va_acc, 4),
            "ped_f1": round(va_per[0], 4), "light_f1": round(va_per[1], 4),
            "medium_f1": round(va_per[2], 4), "heavy_f1": round(va_per[3], 4),
        })
        csv_f.flush()
        elapsed = time.time() - t_e
        print(f"  E{epoch:3d} | {elapsed:5.1f}s | "
              f"train loss={tr_loss:.4f} f1={tr_f1:.3f} | "
              f"val loss={va_loss:.4f} f1={va_f1:.3f} acc={va_acc:.3f} | "
              f"per-class={[round(x, 2) for x in va_per]}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pth")

    csv_f.close()

    print(f"\nLoading best (epoch {best_epoch}, val_f1={best_val_f1:.4f}) for test eval …")
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=dev))
    with torch.no_grad():
        te_loss, te_f1, te_acc, te_per = run_one_pass(model, test_loader, dev, None, type_w)
    print(f"  TEST | loss={te_loss:.4f} f1={te_f1:.4f} acc={te_acc:.4f}  "
          f"per-class={[round(x, 4) for x in te_per]}")

    summary = {
        "config": asdict(cfg),
        "args": vars(args),
        "best_val_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "test_f1": round(te_f1, 4),
        "test_acc": round(te_acc, 4),
        "test_per_class_f1": {
            "pedestrian": round(te_per[0], 4),
            "light":      round(te_per[1], 4),
            "medium":     round(te_per[2], 4),
            "heavy":      round(te_per[3], 4),
        },
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "type_class_weights": [round(w, 4) for w in type_w.cpu().tolist()],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {csv_path}")
    print(f"  Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
