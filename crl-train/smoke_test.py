"""
smoke_test.py — end-to-end pipeline sanity check.

Loads real data (first --max-files parquet files from each split),
runs one forward + backward pass through the full pipeline, and
prints tensor shapes at every stage plus a loss breakdown.

Completes in < 60 s on CPU with the default settings.

Usage:
    python smoke_test.py
    python smoke_test.py --data-dir ../data/parsed/train --val-dir ../data/parsed/val
    python smoke_test.py --sensors seismic           # single modality
    python smoke_test.py --max-files 5 --batch-size 4
"""

import argparse
import sys
from pathlib import Path
import os, shutil, tempfile

import torch

# Ensure crl_vehicle and training packages are importable from crl-train/
sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import (
    SensorDataset,
    MultiHorizonPairDataset,
    collate_single,
    collate_pairs,
)
from crl_vehicle.losses.combined import CombinedLoss
from training.trainer import CRLModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print("=" * 60)


def _check(name, tensor, expected_shape=None):
    ok = (
        "✓"
        if (expected_shape is None or tuple(tensor.shape) == tuple(expected_shape))
        else "✗"
    )
    print(f"  {ok}  {name}: {tuple(tensor.shape)}  dtype={tensor.dtype}")
    if not tensor.isfinite().all():
        print(f"     WARNING: non-finite values detected in {name}!")


def _slim_dataset(
    parquet_dir: str, config: CRLConfig, max_files: int, is_train: bool
) -> SensorDataset:
    """
    Create a SensorDataset but cap the number of parquet files loaded
    by temporarily limiting glob results.
    """

    src = Path(parquet_dir)
    # Select up to max_files per modality (not first N alphabetically,
    # which could be all 'accel' files that the dataset ignores).
    all_parquet = sorted(src.glob("*.parquet"))
    files = []
    for sensor in MODALITIES:
        sensor_files = [
            f for f in all_parquet if f.stem.split("_", 2)[1:2] == [sensor]
        ][:max_files]
        files.extend(sensor_files)
    if not files:
        raise FileNotFoundError(f"No parquet files for modalities {MODALITIES} in {parquet_dir}")

    # Write a temp dir with symlinks to the capped file list
    tmp = Path(tempfile.mkdtemp(prefix="crl_smoke_"))
    try:
        for f in files:
            (tmp / f.name).symlink_to(f.resolve())
        ds = SensorDataset(str(tmp), config, is_train=is_train)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../data_files/parsed/train")
    p.add_argument("--val-dir", default="../data_files/parsed/val")
    p.add_argument(
        "--sensors", nargs="+", default=["audio", "seismic"], choices=MODALITIES
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-files",
        type=int,
        default=6,
        help="Max parquet files to load per split (per sensor)",
    )
    p.add_argument("--device", default=None, help="Force device: cpu / cuda / mps")
    return p.parse_args()


def main():
    args = parse_args()
    print(args)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    sensors = args.sensors or MODALITIES

    # Config: minimal epochs, small model for fast smoke test
    cfg = CRLConfig()
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0  # no multiprocessing — avoids fork issues locally
    cfg.n_epochs = 1
    cfg.d_model = 32  # smaller for speed
    cfg.ssm_layers = 1

    # ----------------------------------------------------------------
    _header("1. Data loading")
    # ----------------------------------------------------------------
    try:
        train_ds = _slim_dataset(args.data_dir, cfg, args.max_files, is_train=True)
        val_ds = _slim_dataset(args.val_dir, cfg, args.max_files, is_train=False)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Have you run:  python old/split_data.py  to create the split dirs?")
        sys.exit(1)

    print(f"  Train windows : {len(train_ds)}")
    print(f"  Val windows   : {len(val_ds)}")

    if len(train_ds) < args.batch_size:
        print(
            f"  WARNING: fewer windows ({len(train_ds)}) than batch_size "
            f"({args.batch_size}). Reduce --batch-size or increase --max-files."
        )
        cfg.batch_size = max(1, len(train_ds))

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_single,
    )
    batch = next(iter(loader))

    print("\n  Batch keys and shapes:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"    {k}: {tuple(v.shape)}")
        else:
            print(f"    {k}: {v}")

    # ----------------------------------------------------------------
    _header("2. Model construction")
    # ----------------------------------------------------------------
    model = CRLModel(cfg, sensors=sensors).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n:,}")

    # ----------------------------------------------------------------
    _header("3. Shape trace: Filterbank → SSM → Encoder")
    # ----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for sensor in sensors:
            x = batch[f"x_{sensor}"].to(device)
            print(f"\n  [{sensor}]")
            _check("  input x", x)

            y = model.filterbanks[sensor](x)
            mod_cfg = cfg.modality_cfg(sensor)
            _check(
                "  filterbank out (B, K*C, T')",
                y,
                (x.shape[0], mod_cfg.filterbank_out_channels, mod_cfg.t_prime),
            )

            h = model.ssms[sensor](y)
            _check(
                "  SSM out (B, T', d_model)",
                h,
                (x.shape[0], mod_cfg.t_prime, cfg.d_model),
            )

            z, mu, lv = model.encoders[sensor](h)
            _check("  z  (B, d_z)", z, (x.shape[0], cfg.d_z))
            _check("  mu (B, d_z)", mu)
            _check("  log_var", lv)

            z_scm = model.scm(z)
            _check("  z_scm (B, d_z)", z_scm, (x.shape[0], cfg.d_z))

            x_hat = model.decoders[sensor](z)
            _check(
                "  decoder x_hat (B, K, T')",
                x_hat,
                (x.shape[0], mod_cfg.n_filters, mod_cfg.t_prime),
            )

    acyc = model.scm.acyclicity_loss()
    print(f"\n  acyclicity_loss = {acyc.item():.4f}  (should decrease toward 0)")

    # ----------------------------------------------------------------
    _header("4. Full forward pass + loss")
    # ----------------------------------------------------------------
    model.train()
    loss_fn = CombinedLoss(cfg)
    loss_fn.update_beta(epoch=0)

    outputs = model.forward_known(batch, device)
    total_loss, metrics = loss_fn(outputs)

    print(f"\n  Total loss: {total_loss.item():.4f}")
    print("\n  Loss breakdown:")
    for k, v in sorted(metrics.items()):
        print(f"    {k:20s}: {v:.4f}")

    if not total_loss.isfinite():
        print(
            "\n  ERROR: non-finite total loss — check loss weights and log_var clamping."
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    _header("5. Backward pass + gradient check")
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    outputs = model.forward_known(batch, device)
    total_loss, _ = loss_fn(outputs)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Check for None / NaN gradients
    bad_grads = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and not p.grad.isfinite().all()
    ]
    no_grads = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    if bad_grads:
        print(f"  WARNING: non-finite gradients in: {bad_grads[:5]}")
    elif no_grads:
        print(f"  NOTE: no gradient for (disconnected?): {no_grads[:5]}")
    else:
        print("  ✓ All gradients finite and present.")

    # ----------------------------------------------------------------
    _header("6. Multi-horizon pair forward (causal temporal training)")
    # ----------------------------------------------------------------
    pair_ds = MultiHorizonPairDataset(train_ds)
    if len(pair_ds) == 0:
        print(
            "  SKIP: no multi-horizon pairs in this slice "
            "(increase --max-files to get pairs)."
        )
    else:
        pair_loader = DataLoader(
            pair_ds,
            batch_size=min(cfg.batch_size, len(pair_ds)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pairs,
        )
        pair_batch = next(iter(pair_loader))
        horizon_ns = pair_batch.get("horizon_n")
        print(
            f"  Pair batch size: {pair_batch['x_seismic_t'].shape[0] if 'x_seismic_t' in pair_batch else 'n/a'}, "
            f"horizon_n range: [{horizon_ns.min().item()}, {horizon_ns.max().item()}]"
            if horizon_ns is not None else ""
        )
        pair_outputs = model.forward_horizon_pair(pair_batch, device)
        pair_loss, pair_m = loss_fn(pair_outputs)
        print(f"  Horizon-pair loss: {pair_loss.item():.4f}")
        if "interv_logits" in pair_outputs:
            _check("  interv_logits", pair_outputs["interv_logits"])

    # ----------------------------------------------------------------
    _header("SMOKE TEST PASSED")
    # ----------------------------------------------------------------
    print("  Pipeline is functional. Ready to run on the training server.")
    print(f"\n  Full training command:")
    sensor_flag = f"--sensors {' '.join(sensors)}" if sensors != MODALITIES else ""
    print(
        f"    python train.py --phase full "
        f"--data-dir {args.data_dir} --val-dir {args.val_dir} {sensor_flag}"
    )


if __name__ == "__main__":
    main()
