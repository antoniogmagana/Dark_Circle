#!/usr/bin/env python3
"""
run_experiments.py — Diagnostic ablation runner.

Runs three CRL experiments in order to isolate the cause of non-convergence
and low probe_type_f1. Each experiment writes its own save-dir under
saved_crl/experiments/<name>/ and appends a summary to
saved_crl/experiments/report.json when it finishes.

Experiments
-----------
exp1_no_tc      : lambda_tc=0, lambda_tc_cross=0
                  Isolates whether the TC loss is destroying type signal.

exp2_no_tc_recon: lambda_tc=0, lambda_tc_cross=0, lambda_recon=0
                  Stacked on exp1: checks if reconstruction further competes.

exp3_no_losses  : lambda_tc=0, lambda_tc_cross=0, lambda_recon=0,
                  lambda_causal=0
                  Pure supervised baseline — only pres + type + inst CE losses.

Usage
-----
    cd crl-train
    python run_experiments.py

    # Limit to a fast sanity check (50 steps/epoch):
    python run_experiments.py --steps-per-epoch 50

    # Only run specific experiments (space-separated names):
    python run_experiments.py --only exp1_no_tc exp2_no_tc_recon

    # Change data paths if different from train.py defaults:
    python run_experiments.py --data-dir ../data_files/parsed/train \\
                              --val-dir  ../data_files/parsed/val

All other hyperparameters (lr, batch size, architecture, etc.) are left at
their CRLConfig defaults so results are directly comparable to the baseline.
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
from crl_vehicle.data.dataset import SensorDataset, collate_single
from crl_vehicle.losses.combined import SupervisedMultiTaskLoss
from training.trainer import CRLModel, Trainer
from training.eval import run_full_eval


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name":        "exp1_no_tc",
        "description": "No TC losses (lambda_tc=0, lambda_tc_cross=0)",
        "overrides": {
            "lambda_tc":       0.0,
            "lambda_tc_cross": 0.0,
        },
    },
    {
        "name":        "exp2_no_tc_recon",
        "description": "No TC or reconstruction losses",
        "overrides": {
            "lambda_tc":       0.0,
            "lambda_tc_cross": 0.0,
            "lambda_recon":    0.0,
        },
    },
    {
        "name":        "exp3_no_losses",
        "description": "Pure supervised baseline (only pres + type + inst CE)",
        "overrides": {
            "lambda_tc":       0.0,
            "lambda_tc_cross": 0.0,
            "lambda_recon":    0.0,
            "lambda_causal":   0.0,
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
    train_ds = SensorDataset(data_dir, cfg, is_train=True)
    val_ds   = SensorDataset(val_dir,  cfg, is_train=False)
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_single,
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


def probe_summary(metrics: dict) -> str:
    return (
        f"pres_f1={metrics.get('probe_pres_f1', 0):.3f}  "
        f"type_f1={metrics.get('probe_type_f1', 0):.3f}  "
        f"inst_f1={metrics.get('probe_inst_f1', 0):.3f}  "
        f"auc={metrics.get('detection_auc', 0):.3f}"
    )


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
    loss_fn = SupervisedMultiTaskLoss(cfg, scm=model.scm).to(device)
    trainer = Trainer(model, loss_fn, cfg, device, save_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    trainer.train_crl(train_loader, val_loader, cfg.n_epochs)
    elapsed = time.time() - t0
    print(f"  CRL training done in {elapsed/60:.1f} min")

    # Load best checkpoint for final evaluation
    best_ckpt = save_dir / "crl_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=True)
        )
        print(f"  Loaded best checkpoint: {best_ckpt}")

    print("  Running final embedding evaluation...")
    final_metrics = run_full_eval(
        model, train_loader, val_loader, device,
        max_batches=cfg.steps_per_epoch,
    )
    print(f"  Final: {probe_summary(final_metrics)}")

    # Save per-experiment summary JSON
    summary = {
        "name":         name,
        "description":  description,
        "overrides":    overrides,
        "elapsed_min":  round(elapsed / 60, 2),
        "final_metrics": final_metrics,
    }
    with open(save_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

BASELINE = {
    "probe_type_f1":    0.197,
    "probe_pres_f1":    0.792,
    "probe_inst_f1":    0.098,
    "detection_auc":    0.882,
    "noise_sep_in_z_veh": 0.400,
}

TARGET = {
    "probe_type_f1":    0.35,
    "noise_sep_in_z_veh": 0.25,
}


def write_report(summaries: list[dict], report_path: Path) -> None:
    report = {
        "baseline": BASELINE,
        "target":   TARGET,
        "experiments": summaries,
        "comparison": [],
    }

    for s in summaries:
        m = s["final_metrics"]
        delta_type_f1 = m.get("probe_type_f1", 0) - BASELINE["probe_type_f1"]
        verdict = "IMPROVED" if delta_type_f1 > 0.05 else (
            "MARGINAL" if delta_type_f1 > 0.01 else "NO_CHANGE"
        )
        report["comparison"].append({
            "name":            s["name"],
            "description":     s["description"],
            "overrides":       s["overrides"],
            "probe_type_f1":   round(m.get("probe_type_f1", 0), 4),
            "probe_pres_f1":   round(m.get("probe_pres_f1", 0), 4),
            "probe_inst_f1":   round(m.get("probe_inst_f1", 0), 4),
            "detection_auc":   round(m.get("detection_auc",  0), 4),
            "delta_type_f1":   round(delta_type_f1, 4),
            "verdict":         verdict,
        })

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    print(f"\n{'=' * 65}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Experiment':<24} {'type_f1':>8} {'Δtype_f1':>9} {'pres_f1':>8} {'auc':>7}  verdict")
    print(f"  {'-'*24} {'-'*8} {'-'*9} {'-'*8} {'-'*7}  -------")
    print(f"  {'baseline (current run)':<24} {BASELINE['probe_type_f1']:>8.3f} {'—':>9} {BASELINE['probe_pres_f1']:>8.3f} {BASELINE['detection_auc']:>7.3f}")
    for c in report["comparison"]:
        print(
            f"  {c['name']:<24} {c['probe_type_f1']:>8.3f} "
            f"{c['delta_type_f1']:>+9.3f} {c['probe_pres_f1']:>8.3f} "
            f"{c['detection_auc']:>7.3f}  {c['verdict']}"
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
                "description":  exp["description"],
                "overrides":    exp["overrides"],
                "error":        str(exc),
                "final_metrics": {},
            })

    report_path = experiments_dir / "report.json"
    write_report(summaries, report_path)


if __name__ == "__main__":
    main()
