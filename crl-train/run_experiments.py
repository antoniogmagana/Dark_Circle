#!/usr/bin/env python3
"""
run_experiments.py — CRL diagnostic ablation runner.

Experiments
-----------
exp1_baseline      : multiscale, noise-type intervention (current behaviour), no aux supervision
exp2_aux_on        : multiscale, noise-type intervention, aux supervision ON
exp3_redesigned    : multiscale, label-change intervention, aux supervision ON
exp4_morlet        : morlet, label-change intervention, aux supervision ON
exp5_interv_strong : multiscale, label-change intervention, lambda_interv=2.0, aux supervision ON

Usage
-----
    python run_experiments.py
    python run_experiments.py --steps-per-epoch 50
    python run_experiments.py --only exp1_baseline exp3_redesigned
    python run_experiments.py --hardware-profile mid
"""

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig, MODALITIES
from crl_vehicle.data.dataset import (
    SensorDataset, ConsecutivePairDataset,
    collate_pairs, collate_single,
)
from training.trainer import CRLModel, Trainer


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name":        "exp1_baseline",
        "description": "Multiscale, noise-type intervention, no aux supervision (current behaviour)",
        "overrides":   {
            "frontend_type":       "multiscale",
            "lambda_interv":       1.0,
            "use_aux_supervision": False,
            "intervention_mode":   "noise_type",
        },
    },
    {
        "name":        "exp2_aux_on",
        "description": "Multiscale, noise-type intervention, aux supervision ON",
        "overrides":   {
            "frontend_type":       "multiscale",
            "lambda_interv":       1.0,
            "use_aux_supervision": True,
            "intervention_mode":   "noise_type",
        },
    },
    {
        "name":        "exp3_redesigned",
        "description": "Multiscale, label-change intervention, aux supervision ON",
        "overrides":   {
            "frontend_type":       "multiscale",
            "lambda_interv":       1.0,
            "use_aux_supervision": True,
            "intervention_mode":   "label_change",
        },
    },
    {
        "name":        "exp4_morlet",
        "description": "Morlet frontend, label-change intervention, aux supervision ON",
        "overrides":   {
            "frontend_type":       "morlet",
            "lambda_interv":       1.0,
            "use_aux_supervision": True,
            "intervention_mode":   "label_change",
        },
    },
    {
        "name":        "exp5_interv_strong",
        "description": "Multiscale, label-change intervention, lambda_interv=2.0, aux supervision ON",
        "overrides":   {
            "frontend_type":       "multiscale",
            "lambda_interv":       2.0,
            "use_aux_supervision": True,
            "intervention_mode":   "label_change",
        },
    },
]


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

HARDWARE_PROFILES = {
    "h100": {"batch_size": 512, "d_model": 128, "n_layers": 4, "num_workers": 12, "steps_per_epoch": None},
    "mid":  {"batch_size": 128, "d_model": 64,  "n_layers": 2, "num_workers": 8,  "steps_per_epoch": None},
    "low":  {"batch_size": 64,  "d_model": 64,  "n_layers": 2, "num_workers": 4,  "steps_per_epoch": 200},
    "cpu":  {"batch_size": 32,  "d_model": 32,  "n_layers": 1, "num_workers": 2,  "steps_per_epoch": 50},
}


def detect_hardware_profile() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 60:
        return "h100"
    elif vram_gb >= 16:
        return "mid"
    else:
        return "low"


def apply_hardware_profile(cfg: CRLConfig, profile_name: str) -> CRLConfig:
    profile = HARDWARE_PROFILES[profile_name]
    cfg.hardware_profile_name = profile_name
    for k, v in profile.items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.n_heads = max(2, cfg.d_model // 32)
    return cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {name} ({vram:.0f}GB VRAM)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("  Device: Apple Silicon MPS")
        return torch.device("mps")
    else:
        print("  Device: CPU (slow)")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def build_pair_loaders(data_dir: str, val_dir: str, cfg: CRLConfig):
    """For CRL pre-training — consecutive pairs."""
    train_ds = ConsecutivePairDataset(SensorDataset(data_dir, cfg, is_train=True))
    val_ds   = ConsecutivePairDataset(SensorDataset(val_dir,  cfg, is_train=False))
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        pin_memory=(cfg.hardware_profile_name != "cpu"),
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
    )


def build_single_loaders(data_dir: str, val_dir: str, cfg: CRLConfig):
    """For downstream head training — single windows."""
    train_ds = SensorDataset(data_dir, cfg, is_train=True)
    val_ds   = SensorDataset(val_dir,  cfg, is_train=False)
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_single,
        pin_memory=(cfg.hardware_profile_name != "cpu"),
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_overrides(cfg: CRLConfig, overrides: dict) -> CRLConfig:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"CRLConfig has no attribute '{k}'")
        setattr(cfg, k, v)
    return cfg


def _best_downstream_f1(save_dir: Path) -> dict:
    results = {}
    for sensor in MODALITIES:
        path = save_dir / f"downstream_metrics_{sensor}.csv"
        if not path.exists():
            continue
        best_pres = best_type = 0.0
        with open(path) as f:
            for row in csv.DictReader(f):
                best_pres = max(best_pres, float(row.get("val_pres_f1", 0)))
                best_type = max(best_type, float(row.get("val_type_f1", 0)))
        results[f"best_pres_f1_{sensor}"] = best_pres
        results[f"best_type_f1_{sensor}"] = best_type
    return results


def _best_crl_elbo(save_dir: Path) -> tuple[float, int]:
    path = save_dir / "crl_metrics.csv"
    best = float("inf")
    converged_epoch = -1
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                val = float(row.get("val_ref_elbo", "inf"))
                if val < best:
                    best = val
                    converged_epoch = int(row.get("epoch", -1))
    return best, converged_epoch


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
    name        = exp["name"]
    description = exp["description"]
    overrides   = exp["overrides"]

    print(f"\n{'=' * 65}")
    print(f"  Experiment: {name}")
    print(f"  {description}")
    print("  Overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
    print("=" * 65)

    save_dir = experiments_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_overrides(deepcopy(base_cfg), overrides)
    cfg.save_dir = str(save_dir)

    pair_train, pair_val     = build_pair_loaders(data_dir, val_dir, cfg)
    single_train, single_val = build_single_loaders(data_dir, val_dir, cfg)

    model   = CRLModel(cfg, sensors=MODALITIES).to(device)
    trainer = Trainer(model, cfg, device, save_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # CRL pre-training
    t0 = time.time()
    trainer.train_crl(pair_train, pair_val, cfg.n_epochs)
    crl_elapsed = time.time() - t0
    print(f"  CRL training done in {crl_elapsed/60:.1f} min")

    # Load best CRL checkpoint before downstream
    best_ckpt = save_dir / "crl_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    # Downstream head training
    t1 = time.time()
    trainer.train_downstream(single_train, single_val, cfg.n_epochs)
    ds_elapsed = time.time() - t1
    print(f"  Downstream training done in {ds_elapsed/60:.1f} min")

    best_elbo, converged_epoch = _best_crl_elbo(save_dir)
    f1_scores = _best_downstream_f1(save_dir)

    summary = {
        "name":                   name,
        "description":            description,
        "overrides":              overrides,
        "crl_elapsed_min":        round(crl_elapsed / 60, 2),
        "downstream_elapsed_min": round(ds_elapsed / 60, 2),
        "best_val_ref_elbo":      best_elbo,
        "crl_converged_epoch":    converged_epoch,
        **f1_scores,
    }
    with open(save_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(summaries: list[dict], report_path: Path) -> None:
    baseline_summary = next((s for s in summaries if s["name"] == "exp1_baseline"), None)
    baseline_f1 = 0.0
    if baseline_summary:
        f1_vals = [v for k, v in baseline_summary.items()
                   if k.startswith("best_pres_f1") or k.startswith("best_type_f1")]
        baseline_f1 = sum(f1_vals) / max(len(f1_vals), 1)

    comparison = []
    for s in summaries:
        f1_vals = [v for k, v in s.items()
                   if k.startswith("best_pres_f1") or k.startswith("best_type_f1")]
        mean_f1 = sum(f1_vals) / max(len(f1_vals), 1)
        delta   = mean_f1 - baseline_f1
        verdict = "IMPROVED" if delta > 0.05 else ("MARGINAL" if delta > 0 else "NO_CHANGE")
        comparison.append({
            "name":               s["name"],
            "description":        s["description"],
            "best_val_ref_elbo":  round(s.get("best_val_ref_elbo", float("inf")), 4),
            "mean_downstream_f1": round(mean_f1, 4),
            "delta_f1":           round(delta, 4),
            "verdict":            verdict,
            **{k: round(v, 4) for k, v in s.items()
               if k.startswith("best_pres_f1") or k.startswith("best_type_f1")},
        })

    report = {"summaries": summaries, "comparison": comparison}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 75}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Experiment':<24} {'ELBO':>8} {'Mean F1':>8} {'dF1':>8}  Verdict")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}  -------")
    for c in comparison:
        print(
            f"  {c['name']:<24} {c['best_val_ref_elbo']:>8.3f} "
            f"{c['mean_downstream_f1']:>8.3f} {c['delta_f1']:>+8.3f}  {c['verdict']}"
        )
    print(f"\n  Report written to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CRL diagnostic ablation runner")
    p.add_argument("--data-dir",         default="../data_files/parsed/train")
    p.add_argument("--val-dir",          default="../data_files/parsed/val")
    p.add_argument("--crl-epochs",       type=int, default=None)
    p.add_argument("--batch-size",       type=int, default=None)
    p.add_argument("--steps-per-epoch",  type=int, default=None)
    p.add_argument("--num-workers",      type=int, default=None)
    p.add_argument("--out-dir",          default="./saved_crl/experiments")
    p.add_argument("--hardware-profile", choices=list(HARDWARE_PROFILES), default=None,
                   help="Override hardware auto-detection")
    p.add_argument("--only",             nargs="+", default=None, metavar="NAME")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    cfg = CRLConfig()

    profile_name = args.hardware_profile or detect_hardware_profile()
    apply_hardware_profile(cfg, profile_name)
    print(f"  Profile: {profile_name} "
          f"(batch={cfg.batch_size}, d_model={cfg.d_model}, "
          f"n_layers={cfg.n_layers}, workers={cfg.num_workers})")

    if args.crl_epochs is not None:
        cfg.n_epochs = args.crl_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    experiments_dir = Path(args.out_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        valid = {e["name"] for e in EXPERIMENTS}
        unknown = set(args.only) - valid
        if unknown:
            print(f"ERROR: unknown names: {unknown}. Valid: {sorted(valid)}")
            sys.exit(1)
        experiments = [e for e in EXPERIMENTS if e["name"] in args.only]

    print(f"\nRunning {len(experiments)} experiment(s):")
    for e in experiments:
        print(f"  - {e['name']}: {e['description']}")

    summaries = []
    for exp in experiments:
        try:
            summary = run_experiment(
                exp, cfg, args.data_dir, args.val_dir,
                device, experiments_dir,
            )
            summaries.append(summary)
        except Exception as exc:
            import traceback
            print(f"\nERROR in {exp['name']}: {exc}")
            traceback.print_exc()
            summaries.append({
                "name": exp["name"], "description": f"{exp['description']} (FAILED)",
                "overrides": exp["overrides"], "error": str(exc),
            })

    write_report(summaries, experiments_dir / "report.json")


if __name__ == "__main__":
    main()
