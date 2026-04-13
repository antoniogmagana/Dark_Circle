"""
CRL Training Entry Point

Usage:
    python train.py --phase crl   --data-dir ../data/parsed/train --val-dir ../data/parsed/val
    python train.py --phase downstream ...
    python train.py --phase full  ...

Pre-requisite: run  python old/split_data.py  first to split the parquet
files into  data/parsed/{train,val,test_iobt}/

Key flags:
    --sensors      : seismic audio (default: both)
    --crl-epochs   : CRL pre-training epochs (default: 100)
    --ds-epochs    : downstream head epochs (default: 50)
    --batch-size   : default 64
    --save-dir     : where to write checkpoints and metrics CSV
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import (
    SensorDataset,
    collate_single,
)
from crl_vehicle.losses.combined import SupervisedMultiTaskLoss
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


def build_loaders(
    data_dir: str,
    val_dir: str,
    config: CRLConfig,
) -> tuple[DataLoader, DataLoader]:
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
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False, **loader_kwargs
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CRL Vehicle Detection Pipeline")
    p.add_argument(
        "--phase", choices=["crl", "downstream", "full"], default="full"
    )
    p.add_argument("--data-dir", default="../data_files/parsed/train")
    p.add_argument("--val-dir",  default="../data_files/parsed/val")
    p.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        choices=MODALITIES,
        help="Sensor modalities to use (default: all)",
    )
    p.add_argument("--crl-epochs", type=int, default=100)
    p.add_argument("--ds-epochs",  type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--num-workers",type=int,   default=None)
    p.add_argument("--save-dir",   default="./saved_crl")
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=100,
        help="Cap gradient steps per epoch (None = full epoch)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device   = get_device()
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

    sensors = args.sensors or MODALITIES
    print(f"Sensors: {sensors}")

    train_loader, val_loader = build_loaders(args.data_dir, args.val_dir, cfg)

    model = CRLModel(cfg, sensors=sensors)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    loss_fn = SupervisedMultiTaskLoss(cfg, scm=model.scm)
    loss_fn.to(device)
    trainer = Trainer(model, loss_fn, cfg, device, save_dir)

    meta_path = save_dir / "meta.json"

    if args.phase in ("crl", "full"):
        trainer.train_crl(train_loader, val_loader, cfg.n_epochs)
        meta = {
            "sensors": sensors,
            "d_pres": cfg.d_pres,
            "d_type": cfg.d_type,
            "d_inst": cfg.d_inst,
            "n_epochs": cfg.n_epochs,
        }
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    if args.phase in ("downstream", "full"):
        ckpt = save_dir / "crl_best.pth"
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
            print(f"Loaded CRL checkpoint from {ckpt}")
        elif args.phase == "downstream":
            raise FileNotFoundError(
                f"No CRL checkpoint at {ckpt}. Run --phase crl first."
            )

        trainer.train_downstream(train_loader, val_loader, args.ds_epochs)


if __name__ == "__main__":
    main()
