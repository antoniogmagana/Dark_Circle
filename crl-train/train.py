"""CRL training entry point."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    SensorDataset, StratifiedPairDataset,
    collate_pairs, collate_single,
    compute_class_weights,
)
from training.trainer import CRLModel, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CRL model")
    p.add_argument("--phase", choices=["crl", "downstream", "full"], default="full")
    p.add_argument("--data-dir",  default="../data_files/parsed/train/")
    p.add_argument("--val-dir",   default="../data_files/parsed/val/")
    p.add_argument("--sensors",   nargs="+", default=["audio", "seismic"])
    p.add_argument("--crl-epochs",  type=int,   default=100)
    p.add_argument("--ds-epochs",   type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--save-dir",    default=None)
    p.add_argument("--frontend",    choices=["multiscale", "morlet"], default="multiscale")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Limit batches per epoch (for smoke tests)")
    p.add_argument("--cache-dir",   default="./saved_crl/cache")
    # Downstream fine-tuning options
    p.add_argument("--finetune-top-n", type=int, default=0,
                   help="Unfreeze top N encoder transformer layers during downstream "
                        "(0 = fully frozen backbone, -1 = unfreeze all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CRLConfig(
        frontend_type=args.frontend,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        n_epochs=args.crl_epochs,
    )

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(args.save_dir or f"saved_crl/{args.frontend}/{run_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    train_ds = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
    val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)

    pres_weight, type_weights = compute_class_weights(train_ds)
    print(f"  Class weights — pres pos_weight: {pres_weight:.3f} | "
          f"type: {[round(w, 3) for w in type_weights.tolist()]}")

    model   = CRLModel(cfg, sensors=args.sensors).to(device)
    trainer = Trainer(model, cfg, device, save_dir)

    if args.phase in ("crl", "full"):
        train_pair = StratifiedPairDataset(train_ds)
        val_pair   = StratifiedPairDataset(val_ds)
        train_loader = DataLoader(
            train_pair, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_pairs, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_pair, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_pairs, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        trainer.train_crl(
            train_loader, val_loader,
            epochs=args.crl_epochs,
            steps_per_epoch=args.steps_per_epoch,
        )

    if args.phase in ("downstream", "full"):
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_single, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_single, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        trainer.train_downstream(
            train_loader, val_loader,
            epochs=args.ds_epochs,
            pres_pos_weight=pres_weight.to(device),
            type_class_weights=type_weights.to(device),
            finetune_top_n=args.finetune_top_n,
        )

    (save_dir / "meta.json").write_text(json.dumps({
        "config":  asdict(cfg),
        "sensors": args.sensors,
    }, indent=2))
    print(f"Done. Artifacts in {save_dir}")


if __name__ == "__main__":
    main()
