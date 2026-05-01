#!/usr/bin/env python3
"""
run_experiments.py — CRL hyperparameter sweep for disentanglement tuning.

Two execution modes:

1. **Hardcoded EXPERIMENTS list (in-process):** shares one preloaded
   SensorDataset across all runs, so the 30 GB load cost is paid once.
   Backward-compatible with the original design.

2. **YAML sweep (subprocess):** each run launches `train.py` in a fresh
   subprocess for GPU memory + crash isolation. Data is shared via the
   on-disk cache (`cache_dir`), not in-memory sharing. Pass --sweep FILE.

Usage
-----
    # In-process (hardcoded list):
    python run_experiments.py
    python run_experiments.py --steps-per-epoch 50
    python run_experiments.py --only baseline_multiscale high_interv
    python run_experiments.py --crl-epochs 30 --ds-epochs 20

    # YAML sweep (subprocess):
    python run_experiments.py --sweep configs/sweeps/frontend_comparison.yaml
    python run_experiments.py --sweep <file> --steps-per-epoch 2 --crl-epochs 1
"""

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    SensorDataset,
    StratifiedPairDataset,
    collate_pairs,
    collate_single,
    compute_class_weights,
)
from training.trainer import CRLModel, Trainer

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each entry: name, description, overrides (real CRLConfig fields only).
# The goal is to understand which settings improve CRL disentanglement and
# downstream classification performance.

EXPERIMENTS = [
    {
        "name": "baseline_multiscale",
        "description": "Default multiscale frontend, default loss weights",
        "overrides": {},
    },
    {
        "name": "baseline_morlet",
        "description": "Morlet frontend, otherwise default",
        "overrides": {"frontend_type": "morlet"},
    },
    {
        "name": "high_interv",
        "description": "Stronger intervention matching signal (lambda_interv=2.0)",
        "overrides": {"lambda_interv": 2.0},
    },
    {
        "name": "low_interv",
        "description": "Weaker intervention signal (lambda_interv=0.5)",
        "overrides": {"lambda_interv": 0.5},
    },
    {
        "name": "more_partners",
        "description": "2 same-type + 2 diff-type + 2 cross-dataset partners",
        "overrides": {
            "n_partners_same_type": 2,
            "n_partners_diff_type": 2,
            "n_partners_cross_ds": 2,
        },
    },
    {
        "name": "larger_free_subspace",
        "description": "Larger free/nuisance subspace (d_free=13 vs 5) — extra capacity for signal variation the causal slots don't model",
        "overrides": {"d_z": 32},
    },
    {
        "name": "aggressive_beta",
        "description": "Faster beta growth and higher KL target for stronger VAE pressure",
        "overrides": {"beta_step": 0.05, "kl_target": 1.0},
    },
]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {name} ({vram:.0f} GB VRAM)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("  Device: Apple Silicon MPS")
        return torch.device("mps")
    print("  Device: CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def apply_overrides(cfg: CRLConfig, overrides: dict) -> CRLConfig:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"CRLConfig has no attribute '{k}'")
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Loader builders (wrap shared datasets — no I/O)
# ---------------------------------------------------------------------------


def build_loaders(train_ds, val_ds, cfg: CRLConfig, sensors: list[str]):
    pin = torch.cuda.is_available()
    kw = {
        "num_workers": cfg.num_workers,
        "pin_memory": pin,
        "persistent_workers": cfg.num_workers > 0,
    }

    pair_train = DataLoader(
        StratifiedPairDataset(train_ds),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        **kw,
    )
    pair_val = DataLoader(
        StratifiedPairDataset(val_ds),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
        **kw,
    )
    single_train = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_single,
        **kw,
    )
    single_val = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_single,
        **kw,
    )
    return pair_train, pair_val, single_train, single_val


# ---------------------------------------------------------------------------
# Metric readers
# ---------------------------------------------------------------------------


def _best_crl_elbo(save_dir: Path) -> tuple[float, int]:
    path = save_dir / "crl_metrics.csv"
    best = float("inf")
    epoch = -1
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                val = float(row.get("val_ref_elbo", "inf"))
                if val < best:
                    best = val
                    epoch = int(row.get("epoch", -1))
    return best, epoch


def _best_downstream(save_dir: Path) -> dict:
    path = save_dir / "downstream_metrics.csv"
    best: dict[str, float] = {
        "val_pres_f1": 0.0,
        "val_pres_acc": 0.0,
        "val_type_f1": 0.0,
        "val_type_acc": 0.0,
        "val_loss": float("inf"),
    }
    if not path.exists():
        return best
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in ("val_pres_f1", "val_pres_acc", "val_type_f1", "val_type_acc"):
                best[k] = max(best[k], float(row.get(k, 0)))
            vl = float(row.get("val_loss", "inf"))
            best["val_loss"] = min(best["val_loss"], vl)
    return best


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    exp: dict,
    base_cfg: CRLConfig,
    train_ds: SensorDataset,
    val_ds: SensorDataset,
    device: torch.device,
    experiments_dir: Path,
    sensors: list[str],
    crl_epochs: int,
    ds_epochs: int,
    steps_per_epoch: int | None,
    pres_weight: torch.Tensor | None = None,
    type_weights: torch.Tensor | None = None,
) -> dict:
    name = exp["name"]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = experiments_dir / name / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  {name}  ({ts})")
    print(f"  {exp['description']}")
    if exp["overrides"]:
        print("  Overrides: " + ", ".join(f"{k}={v}" for k, v in exp["overrides"].items()))
    print("=" * 65)

    cfg = apply_overrides(deepcopy(base_cfg), exp["overrides"])

    pair_train, pair_val, single_train, single_val = build_loaders(train_ds, val_ds, cfg, sensors)

    model = CRLModel(cfg, sensors=sensors).to(device)
    trainer = Trainer(model, cfg, device, save_dir)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    trainer.train_crl(pair_train, pair_val, epochs=crl_epochs, steps_per_epoch=steps_per_epoch)
    crl_elapsed = time.time() - t0
    print(f"  CRL done in {crl_elapsed/60:.1f} min")

    t1 = time.time()
    trainer.train_downstream(
        single_train,
        single_val,
        epochs=ds_epochs,
        pres_pos_weight=pres_weight.to(device) if pres_weight is not None else None,
        type_class_weights=(type_weights.to(device) if type_weights is not None else None),
    )
    ds_elapsed = time.time() - t1
    print(f"  Downstream done in {ds_elapsed/60:.1f} min")

    best_elbo, conv_epoch = _best_crl_elbo(save_dir)
    ds_best = _best_downstream(save_dir)

    summary = {
        "name": name,
        "timestamp": ts,
        "description": exp["description"],
        "overrides": exp["overrides"],
        "crl_elapsed_min": round(crl_elapsed / 60, 2),
        "ds_elapsed_min": round(ds_elapsed / 60, 2),
        "best_val_ref_elbo": round(best_elbo, 4),
        "crl_converged_epoch": conv_epoch,
        **{k: round(v, 4) for k, v in ds_best.items()},
    }
    (save_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(summaries: list[dict], report_path: Path) -> None:
    baseline = next((s for s in summaries if s["name"] == "baseline_multiscale"), summaries[0])
    baseline_pres_f1 = baseline.get("val_pres_f1", 0.0)
    baseline_type_f1 = baseline.get("val_type_f1", 0.0)

    comparison = []
    for s in summaries:
        delta_pres = s.get("val_pres_f1", 0.0) - baseline_pres_f1
        delta_type = s.get("val_type_f1", 0.0) - baseline_type_f1
        verdict = (
            "IMPROVED"
            if (delta_pres + delta_type) > 0.05
            else ("MARGINAL" if (delta_pres + delta_type) > 0 else "NO_CHANGE")
        )
        comparison.append(
            {
                "name": s["name"],
                "description": s["description"],
                "overrides": s.get("overrides", {}),
                "best_val_ref_elbo": s.get("best_val_ref_elbo", float("inf")),
                "val_pres_f1": s.get("val_pres_f1", 0.0),
                "val_type_f1": s.get("val_type_f1", 0.0),
                "val_pres_acc": s.get("val_pres_acc", 0.0),
                "val_type_acc": s.get("val_type_acc", 0.0),
                "delta_pres_f1": round(delta_pres, 4),
                "delta_type_f1": round(delta_type, 4),
                "verdict": verdict,
            }
        )

    report = {"summaries": summaries, "comparison": comparison}
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 80}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'=' * 80}")
    print(
        f"  {'Experiment':<26} {'ELBO':>7} {'PresF1':>7} {'TypeF1':>7} "
        f"{'dPres':>7} {'dType':>7}  Verdict"
    )
    print(f"  {'-'*26} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  -------")
    for c in comparison:
        elbo = c["best_val_ref_elbo"]
        elbo_str = f"{elbo:7.3f}" if elbo != float("inf") else "    inf"
        print(
            f"  {c['name']:<26} {elbo_str} "
            f"{c['val_pres_f1']:7.3f} {c['val_type_f1']:7.3f} "
            f"{c['delta_pres_f1']:+7.3f} {c['delta_type_f1']:+7.3f}  {c['verdict']}"
        )
    print(f"\n  Report: {report_path}")


# ---------------------------------------------------------------------------
# YAML sweep (subprocess mode)
# ---------------------------------------------------------------------------


def load_sweep_yaml(path: Path) -> tuple[dict, list[dict]]:
    """Load a sweep YAML into (base_config, runs).

    Expected schema:
        base_config:           # dict, applied to every run (optional)
          n_epochs: 100
          lr: 3e-4
        runs:                  # list, required
          - name: my_run_1
            overrides:
              frontend_type: multiscale
          - name: my_run_2
            overrides: {frontend_type: morlet_learnable}

    Returns (base_config, runs). Raises with a clear message on missing keys.
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "YAML sweep mode requires PyYAML. Install with: pip install pyyaml"
        ) from e
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Sweep YAML {path} must be a dict at top level.")
    if "runs" not in data:
        raise ValueError(f"Sweep YAML {path} missing required 'runs' list.")
    if not isinstance(data["runs"], list) or not data["runs"]:
        raise ValueError(f"Sweep YAML {path} 'runs' must be a non-empty list.")
    base = data.get("base_config", {}) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Sweep YAML {path} 'base_config' must be a dict.")
    for i, run in enumerate(data["runs"]):
        if not isinstance(run, dict) or "name" not in run:
            raise ValueError(f"Sweep YAML {path} run[{i}] missing 'name'.")
        if not isinstance(run.get("overrides", {}), dict):
            raise ValueError(f"Sweep YAML {path} run[{i}] 'overrides' must be a dict.")
    return base, data["runs"]


# CRLConfig fields that have a dedicated train.py CLI flag. Everything else
# goes through --config-overrides-json to avoid argparse pollution.
_TRAIN_PY_FLAGS = {
    "frontend_type": "--frontend",
    "training_mode": "--training-mode",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "num_workers": "--num-workers",
    "n_epochs": "--crl-epochs",
    "morlet_learnable_w0": None,  # store_true — special-cased below
    "morlet_learnable_lr_mult": "--morlet-learnable-lr-mult",
}


def _build_train_argv(
    python_exe: str,
    train_script: Path,
    merged_cfg: dict,
    run_save_dir: Path,
    extra_cli: dict,
) -> list[str]:
    """Translate a merged config dict into a train.py argv.

    Fields with dedicated flags get their flag. Fields without a flag get
    bundled into --config-overrides-json. Runtime-only flags (--save-dir,
    --phase, --sensors, --steps-per-epoch, --ds-epochs) come from extra_cli.
    """
    argv = [python_exe, str(train_script), "--save-dir", str(run_save_dir)]
    overrides_for_json: dict = {}

    for k, v in merged_cfg.items():
        flag = _TRAIN_PY_FLAGS.get(k, "missing")
        if flag is None and k == "morlet_learnable_w0":
            # store_true flag — only add when True.
            if v:
                argv.append("--morlet-learnable-w0")
        elif flag == "missing":
            overrides_for_json[k] = v
        else:
            argv += [flag, str(v)]

    if overrides_for_json:
        argv += ["--config-overrides-json", json.dumps(overrides_for_json)]

    for flag, val in extra_cli.items():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]
    return argv


def _collect_run_metrics(run_dir: Path) -> dict:
    """Read crl_metrics.csv + downstream_metrics.csv from a completed run
    and return the best-of-run values. Missing files → default-zero."""
    best_elbo, conv_epoch = _best_crl_elbo(run_dir)
    ds_best = _best_downstream(run_dir)
    return {
        "best_val_ref_elbo": round(best_elbo, 4) if best_elbo != float("inf") else None,
        "crl_converged_epoch": conv_epoch,
        **{k: round(v, 4) for k, v in ds_best.items()},
    }


def run_yaml_sweep(
    sweep_path: Path,
    out_dir: Path,
    extra_cli: dict,
    only: list[str] | None,
    cli_base_overrides: dict | None = None,
) -> list[dict]:
    """Execute each YAML run as a subprocess `python train.py ...`.

    Subprocess mode gives GPU memory + crash isolation — a NaN or OOM in one
    run does not poison the next. Data is shared via on-disk cache (see
    train.py --cache-dir), not in-memory.

    Merge order (highest precedence last):
      YAML base_config → cli_base_overrides → per-run overrides.
    cli_base_overrides lets the top-level `run_experiments.py --crl-epochs N`
    override what the sweep YAML specified — "user knows best" signal.

    Returns a list of per-run summaries.
    """
    base, runs = load_sweep_yaml(sweep_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if only:
        valid = {r["name"] for r in runs}
        unknown = set(only) - valid
        if unknown:
            raise ValueError(f"--only: unknown names {sorted(unknown)}. " f"Valid: {sorted(valid)}")
        runs = [r for r in runs if r["name"] in only]

    python_exe = sys.executable
    train_script = Path(__file__).parent / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found at {train_script}")

    summaries: list[dict] = []
    for run in runs:
        name = run["name"]
        run_dir = out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        merged = dict(base)
        if cli_base_overrides:
            merged.update(cli_base_overrides)
        merged.update(run.get("overrides", {}))

        argv = _build_train_argv(
            python_exe=python_exe,
            train_script=train_script,
            merged_cfg=merged,
            run_save_dir=run_dir,
            extra_cli=extra_cli,
        )

        print(f"\n{'=' * 65}")
        print(f"  {name}")
        if run.get("overrides"):
            print(f"  overrides: {run['overrides']}")
        print(f"  argv: {' '.join(shlex.quote(a) for a in argv)}")
        print("=" * 65)

        t0 = time.time()
        try:
            result = subprocess.run(argv, check=False)
            elapsed = time.time() - t0
            if result.returncode != 0:
                summaries.append(
                    {
                        "name": name,
                        "overrides": run.get("overrides", {}),
                        "returncode": result.returncode,
                        "elapsed_min": round(elapsed / 60, 2),
                        "error": f"subprocess exited with code {result.returncode}",
                    }
                )
                print(f"  FAILED: returncode={result.returncode} " f"({elapsed/60:.1f} min)")
                continue
        except KeyboardInterrupt:
            print(f"\n  INTERRUPTED during {name}")
            raise

        metrics = _collect_run_metrics(run_dir)
        summary = {
            "name": name,
            "overrides": run.get("overrides", {}),
            "elapsed_min": round(elapsed / 60, 2),
            "returncode": 0,
            **metrics,
        }
        summaries.append(summary)
        print(f"  OK ({elapsed/60:.1f} min) — {metrics}")

    write_sweep_summary(summaries, out_dir)
    return summaries


def write_sweep_summary(summaries: list[dict], out_dir: Path) -> None:
    """Write sweep summary.csv and summary.json under out_dir."""
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    metric_keys = [
        "best_val_ref_elbo",
        "crl_converged_epoch",
        "val_pres_f1",
        "val_pres_acc",
        "val_type_f1",
        "val_type_acc",
        "val_loss",
        "elapsed_min",
        "returncode",
    ]
    fieldnames = ["name", *metric_keys, "overrides"]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in fieldnames}
            row["overrides"] = json.dumps(s.get("overrides", {}))
            writer.writerow(row)

    summary_json.write_text(json.dumps({"runs": summaries}, indent=2))
    print(f"\nSweep summary: {summary_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CRL hyperparameter sweep")
    p.add_argument("--data-dir", default="../data_files/parsed/train")
    p.add_argument("--val-dir", default="../data_files/parsed/val")
    p.add_argument("--cache-dir", default="./saved_crl/caches/waveform")
    p.add_argument("--out-dir", default="./saved_crl/runs/_archive/experiments_sweep")
    p.add_argument("--crl-epochs", type=int, default=100)
    p.add_argument("--ds-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Limit batches per epoch (for smoke tests)",
    )
    p.add_argument("--only", nargs="+", default=None, metavar="NAME")
    p.add_argument(
        "--sweep",
        default=None,
        metavar="YAML",
        help="Path to a sweep YAML. When set, each run launches "
        "train.py in a subprocess (GPU/crash isolation). When "
        "omitted, the hardcoded EXPERIMENTS list runs in-process.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # YAML sweep mode short-circuits before the in-memory dataset preload —
    # subprocess children each build their own datasets from the shared cache.
    if args.sweep is not None:
        extra_cli = {
            "--data-dir": args.data_dir,
            "--val-dir": args.val_dir,
            "--cache-dir": args.cache_dir,
            "--ds-epochs": args.ds_epochs,
            "--steps-per-epoch": args.steps_per_epoch,
        }
        cli_base: dict = {}
        if args.crl_epochs is not None:
            cli_base["n_epochs"] = args.crl_epochs
        if args.batch_size is not None:
            cli_base["batch_size"] = args.batch_size
        if args.num_workers is not None:
            cli_base["num_workers"] = args.num_workers

        run_yaml_sweep(
            sweep_path=Path(args.sweep),
            out_dir=Path(args.out_dir),
            extra_cli=extra_cli,
            only=args.only,
            cli_base_overrides=cli_base or None,
        )
        return

    device = get_device()

    base_cfg = CRLConfig()
    if args.batch_size is not None:
        base_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        base_cfg.num_workers = args.num_workers

    sensors = ["audio", "seismic"]
    cache_dir = Path(args.cache_dir)
    experiments_dir = Path(args.out_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets ONCE — shared across all experiments
    print("\nPreloading datasets into shared memory …")
    t_load = time.time()
    train_ds = SensorDataset(args.data_dir, base_cfg, is_train=True, cache_dir=cache_dir)
    val_ds = SensorDataset(args.val_dir, base_cfg, is_train=False, cache_dir=cache_dir)
    print(
        f"  Done in {(time.time()-t_load)/60:.1f} min  "
        f"({len(train_ds):,} train / {len(val_ds):,} val windows)"
    )

    pres_weight, type_weights = compute_class_weights(train_ds)
    print(
        f"  Class weights — pres pos_weight: {pres_weight:.3f} | "
        f"type: {[round(w, 3) for w in type_weights.tolist()]}"
    )

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
                exp,
                base_cfg,
                train_ds,
                val_ds,
                device,
                experiments_dir,
                sensors,
                crl_epochs=args.crl_epochs,
                ds_epochs=args.ds_epochs,
                steps_per_epoch=args.steps_per_epoch,
                pres_weight=pres_weight,
                type_weights=type_weights,
            )
            summaries.append(summary)
        except Exception as exc:
            import traceback

            print(f"\nERROR in {exp['name']}: {exc}")
            traceback.print_exc()
            summaries.append(
                {
                    "name": exp["name"],
                    "description": f"{exp['description']} (FAILED)",
                    "overrides": exp["overrides"],
                    "error": str(exc),
                }
            )

    write_report(summaries, experiments_dir / "report.json")


if __name__ == "__main__":
    main()
