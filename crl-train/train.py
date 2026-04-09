"""
CRL Training Entry Point

Usage:
    python train.py --phase crl   --data-dir ../data/parsed/train --val-dir ../data/parsed/val
    python train.py --phase downstream ...
    python train.py --phase full  ...
    python train.py --phase eval  ...    # diagnostic eval only

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
    MultiHorizonPairDataset,
    collate_single,
    collate_pairs,
)
from crl_vehicle.losses.combined import CombinedLoss
from training.trainer import CRLModel, Trainer
from training.eval import run_full_eval


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
    include_pairs: bool = True,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    train_ds = SensorDataset(data_dir, config, is_train=True)
    val_ds = SensorDataset(val_dir, config, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_single,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_single,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True,
    )

    pair_loader = None
    if include_pairs:
        pair_ds = MultiHorizonPairDataset(train_ds)
        if len(pair_ds) > 0:
            pair_loader = DataLoader(
                pair_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_pairs,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=4,
                persistent_workers=True,
            )
            print(f"  Pair dataset: {len(pair_ds)} multi-horizon pairs (n=1..{config.n_horizons}).")
        else:
            print(
                "  Warning: no multi-horizon pairs found — unknown curriculum disabled."
            )

    return train_loader, pair_loader, val_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CRL Vehicle Detection Pipeline")
    p.add_argument(
        "--phase", choices=["crl", "downstream", "full", "eval"], default="full"
    )
    p.add_argument("--data-dir", default="../data_files/parsed/train")
    p.add_argument("--val-dir", default="../data_files/parsed/val")
    p.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        choices=MODALITIES,
        help="Sensor modalities to use (default: all)",
    )
    p.add_argument("--crl-epochs", type=int, default=None)
    p.add_argument("--ds-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--save-dir", default="./saved_crl")
    p.add_argument(
        "--ssm-backend", default="transformer", choices=["transformer", "mamba"]
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    save_dir = Path(args.save_dir)

    # Build config (apply CLI overrides)
    cfg = CRLConfig()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    if args.num_workers:
        cfg.num_workers = args.num_workers
    if args.crl_epochs:
        cfg.n_epochs = args.crl_epochs
    cfg.save_dir = str(save_dir)

    sensors = args.sensors or MODALITIES
    print(f"Sensors: {sensors}")

    # Build data loaders
    train_loader, pair_loader, val_loader = build_loaders(
        args.data_dir, args.val_dir, cfg, include_pairs=True
    )

    # Build model
    model = CRLModel(cfg, sensors=sensors)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    loss_fn = CombinedLoss(cfg)
    trainer = Trainer(model, loss_fn, cfg, device, save_dir)

    meta_path = save_dir / "meta.json"

    if args.phase in ("crl", "full"):
        trainer.train_crl(train_loader, pair_loader, val_loader, cfg.n_epochs)
        meta = {"sensors": sensors, "d_z": cfg.d_z, "n_epochs": cfg.n_epochs}
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

        for sensor, head in model.det_heads.items():
            torch.save(
                head.state_dict(),
                save_dir / f"det_head_{sensor}_final.pth",
            )

    if args.phase == "eval":
        ckpt = save_dir / "crl_best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        primary = "seismic" if "seismic" in sensors else sensors[0]
        metrics = run_full_eval(
            model, train_loader, val_loader, device, primary_sensor=primary
        )
        print("\n=== Evaluation ===")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")
        with open(save_dir / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
