#!/usr/bin/env python3
"""
run_experiments.py — CRL diagnostic ablation runner.

Runs CRL experiments varying frontend architecture and loss weights to
understand their effect on representation quality.

Experiments
-----------
exp1_baseline   : multiscale frontend, default intervention loss weight (1.0)

exp2_interv_up  : multiscale frontend, stronger intervention loss (2.0)
                  Tests if stronger causal pressure improves disentanglement.

exp3_morlet     : morlet wavelet frontend, default intervention loss (1.0)
                  Tests late-fusion architecture vs early-fusion baseline.

Usage
-----
    cd crl-train
    python run_experiments.py

    # Limit to a fast sanity check (50 steps/epoch):
    python run_experiments.py --steps-per-epoch 50

    # Only run specific experiments (space-separated names):
    python run_experiments.py --only exp1_baseline exp2_tc_on

    # Change data paths if different from train.py defaults:
    python run_experiments.py --data-dir ../data_files/parsed/train \\
                              --val-dir  ../data_files/parsed/val

All other hyperparameters (lr, batch size, architecture, etc.) are left at
their CRLConfig defaults so results are directly comparable across experiments.
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import SensorDataset, ConsecutivePairDataset, collate_pairs
from training.trainer import CRLModel, Trainer


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name":        "exp1_baseline",
        "description": "Multiscale frontend, lambda_interv=1.0",
        "overrides":   {"frontend_type": "multiscale", "lambda_interv": 1.0},
    },
    {
        "name":        "exp2_interv_up",
        "description": "Multiscale frontend, stronger intervention loss",
        "overrides": {
            "frontend_type": "multiscale",
            "lambda_interv": 2.0,
        },
    },
    {
        "name":        "exp3_morlet",
        "description": "Morlet wavelet frontend, lambda_interv=1.0",
        "overrides": {
            "frontend_type": "morlet",
            "lambda_interv": 1.0,
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("  Device: Apple Silicon MPS")
    else:
        d = torch.device("cpu")
        print("  Device: CPU (slow)")
    return d


def build_loaders(data_dir: str, val_dir: str, cfg: CRLConfig):
    train_ds = ConsecutivePairDataset(SensorDataset(data_dir, cfg, is_train=True))
    val_ds   = ConsecutivePairDataset(SensorDataset(val_dir,  cfg, is_train=False))
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
    )


def apply_overrides(cfg: CRLConfig, overrides: dict) -> CRLConfig:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"CRLConfig has no attribute '{k}'")
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp: dict,
    base_cfg: CRLConfig,
    data_dir: str,
    val_dir: str,
    device: torch.device,
    experiments_dir: Path,
) -> dict:
    """
    Run one experiment end-to-end.

    Returns a summary dict that is appended to the report.
    """
    name        = exp["name"]
    description = exp["description"]
    overrides   = exp["overrides"]

    print(f"\n{'=' * 65}")
    print(f"  Experiment: {name}")
    print(f"  {description}")
    overrides_str = "  Overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items())
    print(overrides_str)
    print("=" * 65)

    save_dir = experiments_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_overrides(deepcopy(base_cfg), overrides)
    cfg.save_dir = str(save_dir)

    train_loader, val_loader = build_loaders(data_dir, val_dir, cfg)

    model   = CRLModel(cfg, sensors=MODALITIES).to(device)
    trainer = Trainer(model, cfg, device, save_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    trainer.train_crl(train_loader, val_loader, cfg.n_epochs)
    elapsed = time.time() - t0
    print(f"  CRL training done in {elapsed/60:.1f} min")

    # The trainer saves crl_metrics.csv, which can be analyzed by validate_run.py
    # We extract the best validation loss from the CSV for the report.
    best_val_loss = float('inf')
    metrics_path = save_dir / "crl_metrics.csv"
    if metrics_path.exists():
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                best_val_loss = min(best_val_loss, float(row.get("val_ref_elbo", 'inf')))

    # Save per-experiment summary JSON
    summary = {
        "name":         name,
        "description":  description,
        "overrides":    overrides,
        "elapsed_min":  round(elapsed / 60, 2),
        "best_val_ref_elbo": best_val_loss,
    }
    with open(save_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

BASELINE = {
    "best_val_ref_elbo": float('inf'),   # update after first full training run
}

TARGET = {} # Not used for now


def write_report(summaries: list[dict], report_path: Path) -> None:
    report = {
        "baseline": BASELINE,
        "target":   TARGET,
        "experiments": summaries,
        "comparison": [],
    }

    for s in summaries:
        val_loss = s.get("best_val_ref_elbo", float('inf'))
        delta_loss = val_loss - BASELINE["best_val_ref_elbo"]
        verdict = "IMPROVED" if delta_loss < -0.1 else (
            "MARGINAL" if delta_loss < 0 else "NO_CHANGE"
        )
        report["comparison"].append({
            "name":          s["name"],
            "description":   s["description"],
            "overrides":     s["overrides"],
            "best_val_ref_elbo": round(val_loss, 4),
            "delta_loss": round(delta_loss, 4),
            "verdict":       verdict,
        })

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    print(f"\n{'=' * 65}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Experiment':<24} {'Best Val ELBO':>15} {'Δ ELBO':>9}  verdict")
    print(f"  {'-'*24} {'-'*15} {'-'*9}  -------")
    print(f"  {'baseline (current run)':<24} {BASELINE['best_val_ref_elbo']:>15.3f} {'—':>9}")
    for c in report["comparison"]:
        print(
            f"  {c['name']:<24} {c['best_val_ref_elbo']:>15.3f} "
            f"{c['delta_loss']:>+9.3f}  {c['verdict']}"
        )
    print(f"\n  Report written to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CRL diagnostic ablation runner")
    p.add_argument("--data-dir",         default="../data_files/parsed/train")
    p.add_argument("--val-dir",          default="../data_files/parsed/val")
    p.add_argument("--crl-epochs",       type=int,   default=None,
                   help="CRL epochs per experiment (default: CRLConfig.n_epochs=100)")
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--steps-per-epoch",  type=int,   default=None,
                   help="Cap gradient steps/epoch for fast iteration")
    p.add_argument("--num-workers",      type=int,   default=None)
    p.add_argument("--out-dir",          default="./saved_crl/experiments",
                   help="Root dir for experiment outputs")
    p.add_argument("--only",             nargs="+",  default=None,
                   metavar="NAME",
                   help="Run only these experiment names (space-separated)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Base config — all experiments inherit these defaults
    cfg = CRLConfig()
    if args.crl_epochs:
        cfg.n_epochs = args.crl_epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.steps_per_epoch:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.num_workers:
        cfg.num_workers = args.num_workers

    experiments_dir = Path(args.out_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Select experiments
    experiments = EXPERIMENTS
    if args.only:
        valid_names = {e["name"] for e in EXPERIMENTS}
        unknown = set(args.only) - valid_names
        if unknown:
            print(f"ERROR: unknown experiment names: {unknown}")
            print(f"Valid names: {sorted(valid_names)}")
            sys.exit(1)
        experiments = [e for e in EXPERIMENTS if e["name"] in args.only]

    print(f"\nRunning {len(experiments)} experiment(s):")
    for e in experiments:
        print(f"  • {e['name']}: {e['description']}")

    summaries = []
    for exp in experiments:
        try:
            summary = run_experiment(
                exp, cfg, args.data_dir, args.val_dir,
                device, experiments_dir,
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"\nERROR in {exp['name']}: {exc}")
            import traceback
            traceback.print_exc()
            summaries.append({
                "name":         exp["name"],
                "description":  f"{exp['description']} (FAILED)",
                "overrides":    exp["overrides"],
                "error":        str(exc),
                "final_metrics": {},
            })

    report_path = experiments_dir / "report.json"
    write_report(summaries, report_path)


if __name__ == "__main__":
    main()
