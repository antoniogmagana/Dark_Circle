"""
CRL Training Entry Point

Usage:
    python train.py --phase crl        --data-dir ../data/parsed/train --val-dir ../data/parsed/val
    python train.py --phase downstream --data-dir ../data/parsed/train --val-dir ../data/parsed/val
    python train.py --phase full       --data-dir ../data/parsed/train --val-dir ../data/parsed/val

Key flags:
    --sensors      : seismic audio (default: both)
    --crl-epochs   : CRL pre-training epochs (default: 100)
    --ds-epochs    : downstream head epochs (default: 50)
    --batch-size   : default 64
    --save-dir     : where to write checkpoints and metrics CSV
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import SensorDataset, ConsecutivePairDataset, collate_pairs, collate_single
from training.trainer import CRLModel, Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("GPU: Apple Silicon MPS")
    else:
        d = torch.device("cpu")
        print("WARNING: No GPU — using CPU.")
    return d


def build_crl_loaders(
    data_dir: str,
    val_dir: str,
    config: CRLConfig,
) -> tuple[DataLoader, DataLoader]:
    """ConsecutivePairDataset loaders for CRL pre-training."""
    train_ds = ConsecutivePairDataset(SensorDataset(data_dir, config, is_train=True))
    val_ds   = ConsecutivePairDataset(SensorDataset(val_dir,  config, is_train=False))

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs),
    )


def build_downstream_loaders(
    data_dir: str,
    val_dir: str,
    config: CRLConfig,
) -> tuple[DataLoader, DataLoader]:
    """Single-window SensorDataset loaders for downstream head training."""
    train_ds = SensorDataset(data_dir, config, is_train=True)
    val_ds   = SensorDataset(val_dir,  config, is_train=False)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_single,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CRL Vehicle Detection Pipeline")
    p.add_argument("--phase", choices=["crl", "downstream", "full"], default="full")
    p.add_argument("--data-dir", default="../data_files/parsed/train")
    p.add_argument("--val-dir",  default="../data_files/parsed/val")
    p.add_argument("--sensors", nargs="+", default=None, choices=MODALITIES)
    p.add_argument("--crl-epochs",   type=int,   default=100)
    p.add_argument("--ds-epochs",    type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--num-workers",  type=int,   default=8)
    p.add_argument("--save-dir",     default="./saved_crl")
    p.add_argument("--frontend",     type=str, default="multiscale", choices=["multiscale", "morlet"],
                   help="Frontend architecture to use: 'multiscale' (early) or 'morlet' (late).")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    return p.parse_args()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args    = parse_args()
    seed_everything(42)
    device  = get_device()
    save_dir = Path(args.save_dir)

    cfg = CRLConfig()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.num_workers:
        cfg.num_workers = args.num_workers
    if args.crl_epochs:
        cfg.n_epochs = args.crl_epochs
    if args.steps_per_epoch:
        cfg.steps_per_epoch = args.steps_per_epoch
    cfg.save_dir = str(save_dir)
    cfg.frontend_type = args.frontend

    sensors = args.sensors or MODALITIES
    print(f"Sensors: {sensors}")

    model = CRLModel(cfg, sensors=sensors)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    trainer = Trainer(model, cfg, device, save_dir)

    if args.phase in ("crl", "full"):
        crl_train_loader, crl_val_loader = build_crl_loaders(args.data_dir, args.val_dir, cfg)
        trainer.train_crl(crl_train_loader, crl_val_loader, cfg.n_epochs)

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "meta.json", "w") as f:
            json.dump({
                "sensors": sensors,
                "d_z": 10,
                "n_epochs": cfg.n_epochs,
                "frontend": args.frontend,
            }, f, indent=2)

    if args.phase in ("downstream", "full"):
        ckpt = save_dir / "crl_best.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f"Loaded CRL checkpoint from {ckpt}")
        elif args.phase == "downstream":
            raise FileNotFoundError(f"No CRL checkpoint at {ckpt}. Run --phase crl first.")

        ds_train_loader, ds_val_loader = build_downstream_loaders(args.data_dir, args.val_dir, cfg)
        trainer.train_downstream(ds_train_loader, ds_val_loader, args.ds_epochs)


if __name__ == "__main__":
    main()
