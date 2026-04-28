#!/usr/bin/env python3
"""
plot_run.py — diagnostic plots for a single CRL run.

Generates up to four PNG+PDF figures per run:

    training_curves   — recon / KL / total / aux F1 / ref_elbo over epochs
    beta_schedule     — beta and beta_event annotations over epochs
    morlet_freqs      — learnable Morlet frequency drift (learnable runs only)
    downstream_curves — probe train/val F1 over epochs (if downstream exists)

Usage
-----
    python plot_run.py saved_crl/runs/multiscale/vae/v3_lowfreq
    python plot_run.py saved_crl/runs/multiscale/vae/v3_lowfreq --out /tmp/plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crl_vehicle import analysis as A


# --------------------------------------------------------------------------
# Individual figures
# --------------------------------------------------------------------------

def plot_training_curves(ts: dict, out: Path, run_name: str) -> None:
    """6-panel grid of training curves. Falls back silently for missing cols."""
    panels = [
        ("train_recon",       "val_recon",         "Reconstruction"),
        ("train_raw_kl",      "val_raw_kl",        "Raw KL"),
        ("train_total",       "val_total",         "Total loss"),
        ("train_aux_pres_f1", "val_aux_pres_f1",   "Aux presence F1"),
        ("train_aux_type_f1", "val_aux_type_f1",   "Aux type F1"),
        (None,                "val_ref_elbo",      "Val ref_ELBO"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (train_k, val_k, title) in zip(axes.flat, panels):
        epochs = ts.get("epoch", [])
        if train_k and ts.get(train_k):
            ax.plot(epochs[:len(ts[train_k])], ts[train_k], label="train", color="#1f77b4")
        if val_k and ts.get(val_k):
            ax.plot(epochs[:len(ts[val_k])], ts[val_k], label="val", color="#ff7f0e")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if train_k or val_k:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{run_name} — training curves")
    fig.tight_layout()
    _save(fig, out)


def plot_beta_schedule(ts: dict, out: Path, run_name: str) -> None:
    epochs = ts.get("epoch", [])
    betas  = ts.get("beta", [])
    if not epochs or not betas:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs[:len(betas)], betas, color="#2ca02c", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("beta")
    ax.set_title(f"{run_name} — beta schedule")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out)


def plot_morlet_freq_drift(run_dir: Path, out: Path, run_name: str) -> None:
    """Per-sensor frequency drift, one line per filter. Log-y since
    frequencies span 1+ order of magnitude."""
    history = A.load_morlet_freq_history(run_dir)
    if history is None:
        return
    n_sensors = len(history)
    fig, axes = plt.subplots(1, n_sensors, figsize=(6 * n_sensors, 5), squeeze=False)
    for ax, (sensor, filters) in zip(axes[0], history.items()):
        cmap = plt.get_cmap("viridis")
        for idx, freqs in sorted(filters.items()):
            color = cmap(idx / max(len(filters) - 1, 1))
            ax.plot(freqs, color=color, alpha=0.7, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{sensor} — learned center frequencies")
        ax.set_xlabel("epoch")
        ax.set_ylabel("freq (Hz)")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"{run_name} — learnable Morlet frequency drift")
    fig.tight_layout()
    _save(fig, out)


def plot_downstream_curves(run_dir: Path, out: Path, run_name: str) -> None:
    ts = A.load_downstream_timeseries(run_dir)
    if not ts:
        return
    panels = [
        ("train_pres_f1", "val_pres_f1", "Presence F1"),
        ("train_type_f1", "val_type_f1", "Type F1"),
        ("train_loss",    "val_loss",    "Loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, (train_k, val_k, title) in zip(axes, panels):
        epochs = ts.get("epoch", [])
        if ts.get(train_k):
            ax.plot(epochs[:len(ts[train_k])], ts[train_k], label="train", color="#1f77b4")
        if ts.get(val_k):
            ax.plot(epochs[:len(ts[val_k])], ts[val_k], label="val", color="#ff7f0e")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{run_name} — downstream probe")
    fig.tight_layout()
    _save(fig, out)


# --------------------------------------------------------------------------
# Shared save utility
# --------------------------------------------------------------------------

def _save(fig, out_stem: Path) -> None:
    """Save to PNG + PDF, reporting both."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=120)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path,
                   help="Path to a single run dir under saved_crl/runs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir. Default: saved_crl/analysis/plots/<run_name>/")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    if not (run_dir / "crl" / "meta.json").exists():
        print(f"{run_dir}/crl/meta.json not found", file=sys.stderr)
        return 1

    out_dir = args.out or (Path("saved_crl/analysis/plots") / run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name

    ts = A.load_crl_timeseries(run_dir)

    plot_training_curves(ts, out_dir / "training_curves", run_name)
    plot_beta_schedule(ts, out_dir / "beta_schedule", run_name)
    plot_morlet_freq_drift(run_dir, out_dir / "morlet_freq_drift", run_name)
    plot_downstream_curves(run_dir, out_dir / "downstream_curves", run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
