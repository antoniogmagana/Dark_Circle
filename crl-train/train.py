"""CRL training entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    SensorDataset, StratifiedPairDataset,
    collate_pairs, collate_single,
)
from training.trainer import CRLModel, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CRL model")
    p.add_argument("--phase", choices=["crl", "downstream", "full"], default="full")
    p.add_argument("--data-dir", required=True, help="Training parquet directory", default="../data_files/parsed/train/")
    p.add_argument("--val-dir",  required=True, help="Validation parquet directory", default="../data_files/parsed/val/")
    p.add_argument("--sensors", nargs="+", default=["audio", "seismic"])
    p.add_argument("--crl-epochs",  type=int, default=100)
    p.add_argument("--ds-epochs",   type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-dir",    default=None)
    p.add_argument("--frontend",    choices=["multiscale", "morlet"], default="multiscale")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Limit batches per epoch (for smoke tests)")
    p.add_argument("--cache-dir",   default=None, help="Dataset disk cache directory")
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

    save_dir = Path(args.save_dir or f"saved_crl/{args.frontend}")
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    model   = CRLModel(cfg, sensors=args.sensors).to(device)
    trainer = Trainer(model, cfg, device, save_dir)

    if args.phase in ("crl", "full"):
        train_ds   = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
        val_ds     = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)
        train_pair = StratifiedPairDataset(train_ds)
        val_pair   = StratifiedPairDataset(val_ds)
        train_loader = DataLoader(
            train_pair, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_pairs, pin_memory=True,
        )
        val_loader = DataLoader(
            val_pair, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_pairs, pin_memory=True,
        )
        trainer.train_crl(
            train_loader, val_loader,
            epochs=args.crl_epochs,
            steps_per_epoch=args.steps_per_epoch,
        )

    if args.phase in ("downstream", "full"):
        train_ds = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
        val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_single, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_single, pin_memory=True,
        )
        trainer.train_downstream(train_loader, val_loader, epochs=args.ds_epochs)

    meta = {
        "sensors":   args.sensors,
        "d_z":       cfg.d_z,
        "n_epochs":  cfg.n_epochs,
        "frontend":  cfg.frontend_type,
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Artifacts in {save_dir}")


if __name__ == "__main__":
    main()
