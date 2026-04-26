#!/usr/bin/env python3
"""One-shot diagnostic: CRL training + all probes + filtered evals.

Runs the full B1 + B2 + B3 diagnostic pipeline in a single process, reusing one
set of preloaded datasets (avoids re-paying the ~30 GB load cost per phase).

Phases
------
1. CRL pre-training (skipped with --crl-run-dir)
   Emits crl_best.pth (val_ref_elbo) and crl_best_aux_type.pth (val_aux_type_f1).
2. B1 — downstream probes: 3 probe modes × 2 checkpoints = 6 runs.
3. B2 — test evals: each downstream on {full, focal-only, iobt-only} = 18 evals.
4. Consolidated report.json / report.md at the run root.

Usage
-----
    # Full fresh pipeline (~2 hours)
    python run_full_diagnostic.py

    # Reuse an existing CRL run (skips phase 1)
    python run_full_diagnostic.py \\
        --crl-run-dir saved_crl/experiments/baseline_multiscale/2026-04-21_02-22-50

    # Resume an interrupted run (idempotent on completed sub-runs)
    python run_full_diagnostic.py --out-dir saved_crl/oneshot/2026-04-21_14-00-00 \\
        --skip-existing
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    SensorDataset, StratifiedPairDataset,
    collate_pairs, collate_single,
    compute_class_weights,
)
from training.trainer import CRLModel, Trainer
from eval import (
    run_inference, binary_metrics, multiclass_metrics,
    recalibrated_binary_metrics, recalibrated_multiclass_metrics,
    _plot_binary_confusion, _plot_confusion_matrix,
    N_TYPE_CLASSES, IDX_TO_CLASS,
)


# ---------------------------------------------------------------------------
# Probe × checkpoint fan-out
# ---------------------------------------------------------------------------

PROBE_MODES = ("linear_ztype", "mlp_ztype", "linear_fullz")
PROBE_MODES_DISENTANGLED = ("linear_signal", "linear_fullz")


def _probe_modes_for(cfg: CRLConfig) -> tuple[str, ...]:
    """Pick the probe set that matches the latent partition the run produced.

    Disentangled runs use linear_signal (z[0:d_signal]) and linear_fullz only —
    the legacy mlp_ztype/linear_ztype probes slice z[4:10] (CausalLatentSpace.
    D_TYPE), which is meaningless under the 2-block partition. linear_fullz
    is the upper bound across all dims and is partition-agnostic."""
    if cfg.training_mode == "disentangled":
        return PROBE_MODES_DISENTANGLED
    return PROBE_MODES
CKPT_NAMES  = ("crl_best.pth", "crl_best_aux_type.pth")
EVAL_SPLITS = (
    ("full",  None),             # no filter — full test set
    ("focal", ["focal"]),        # ID vehicles (bicycle2, mustang0528)
    ("iobt",  ["iobt"]),         # OOD vehicles (polaris, warhog, silverado)
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir",   default="../data_files/parsed/train/")
    p.add_argument("--val-dir",    default="../data_files/parsed/val/")
    p.add_argument("--test-dir",   default="../data_files/parsed/test/")
    p.add_argument("--cache-dir",  default="./saved_crl/cache")
    p.add_argument("--out-dir",    default=None,
                   help="Root output dir. Defaults to saved_crl/oneshot/<timestamp>/")
    p.add_argument("--crl-run-dir", default=None,
                   help="Existing CRL run dir to reuse. Skips phase 1. Must contain "
                        "meta.json and at least one of {crl_best.pth, crl_best_aux_type.pth}.")
    p.add_argument("--crl-epochs", type=int, default=100)
    p.add_argument("--ds-epochs",  type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Cap batches per CRL epoch (smoke tests)")
    p.add_argument("--frontend",
                   choices=["multiscale", "morlet", "morlet_per_sensor", "morlet_fused",
                            "morlet_learnable", "morlet_learnable_fused"],
                   default="multiscale")
    p.add_argument("--morlet-use-phase", action="store_true",
                   help="Morlet variants emit [log_power, cos_phase, sin_phase] "
                        "→ 3× channels. Preserves phase/onset structure.")
    p.add_argument("--morlet-learnable-w0", action="store_true",
                   help="Make per-filter w0 learnable (only applies to "
                        "morlet_learnable / morlet_learnable_fused).")
    p.add_argument("--morlet-learnable-lr-mult", type=float, default=0.1,
                   help="LR multiplier for learnable Morlet params relative to "
                        "backbone LR (default 0.1).")
    p.add_argument("--prior-type", choices=["standard", "conditional"],
                   default="standard",
                   help="Prior over z. 'standard'=N(0,I); 'conditional'=iVAE "
                        "(label-conditioned MLP → (μ, logσ²)). Conditional "
                        "gives identifiability under label variation.")
    p.add_argument("--training-mode", choices=["vae", "contrastive", "disentangled"],
                   default="vae",
                   help="'vae' (default) = ELBO + aux heads + intervention "
                        "matching. 'contrastive' = NT-Xent over stratified "
                        "partners (no decoder/KL/aux during CRL). 'disentangled' "
                        "= ELBO + 2-block latent (signal/env) with cross-modal "
                        "alignment, env temporal stability, and signal "
                        "intervention invariance losses. Downstream probes "
                        "still run post-hoc for all modes.")
    p.add_argument("--sensors", nargs="+", default=["audio", "seismic"])
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip sub-runs that already have their completion marker.")
    p.add_argument("--recalibrate", action="store_true",
                   help="Write target-prior-calibrated metrics alongside raw metrics "
                        "in each phase_evals report (diagnostic only).")
    p.add_argument("--use-id-split", action="store_true",
                   help="Use the in-distribution split schema (DATASET_VEHICLE_MAP "
                        "markers under id_root). When set, --data-dir/--val-dir/"
                        "--test-dir are ignored and Phase 2/3 read from the same "
                        "split as Phase 1.")
    p.add_argument("--id-root", default="../data_files/parsed/",
                   help="Parent dir containing train/, val/, test/. Used only "
                        "when --use-id-split is set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 — CRL
# ---------------------------------------------------------------------------

def phase_crl(
    cfg: CRLConfig,
    train_ds: SensorDataset,
    val_ds:   SensorDataset,
    device:   torch.device,
    sensors:  list[str],
    crl_dir:  Path,
    crl_epochs: int,
    steps_per_epoch: int | None,
    skip_existing: bool,
) -> Path:
    """Run CRL pre-training, emit crl_best.pth + crl_best_aux_type.pth + meta.json.

    Returns the path to crl_dir. Idempotent: if both checkpoints + meta.json exist
    and --skip-existing is set, no training is done.
    """
    crl_dir.mkdir(parents=True, exist_ok=True)
    meta_path = crl_dir / "meta.json"
    ref_ckpt  = crl_dir / "crl_best.pth"
    aux_ckpt  = crl_dir / "crl_best_aux_type.pth"
    # Contrastive runs emit only crl_best.pth; VAE runs emit both.
    required_ckpts = (ref_ckpt,) if cfg.training_mode == "contrastive" else (ref_ckpt, aux_ckpt)
    done_markers_present = meta_path.exists() and all(p.exists() for p in required_ckpts)

    if skip_existing and done_markers_present:
        print(f"  [skip] CRL outputs already present in {crl_dir}")
        return crl_dir

    print(f"\n{'=' * 72}\n  PHASE 1 — CRL pre-training ({crl_epochs} epochs)\n{'=' * 72}")

    model   = CRLModel(cfg, sensors=sensors).to(device)
    trainer = Trainer(model, cfg, device, crl_dir)

    pair_train = DataLoader(
        StratifiedPairDataset(train_ds), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    pair_val = DataLoader(
        StratifiedPairDataset(val_ds), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    t0 = time.time()
    trainer.train_crl(pair_train, pair_val,
                      epochs=crl_epochs, steps_per_epoch=steps_per_epoch)
    elapsed_min = (time.time() - t0) / 60
    print(f"  CRL done in {elapsed_min:.1f} min")

    crl_meta: dict = {
        "config":  asdict(cfg),
        "sensors": sensors,
        "crl_elapsed_min": round(elapsed_min, 2),
    }
    derived = getattr(model, "_morlet_derived_params", None)
    if derived:
        crl_meta["morlet_derived_params"] = derived
    meta_path.write_text(json.dumps(crl_meta, indent=2))
    return crl_dir


def ensure_crl_dir_usable(crl_dir: Path) -> dict:
    """Validate a reused CRL run directory and return its meta.json contents.

    Accepts two metadata formats:
      - meta.json                  : produced by train.py (contains config, sensors)
      - experiment_summary.json    : produced by run_experiments.py (contains overrides
                                     relative to CRLConfig defaults)

    Raises if neither exists or no CRL checkpoint is present.
    """
    has_any = any((crl_dir / c).exists() for c in CKPT_NAMES)
    if not has_any:
        raise FileNotFoundError(
            f"Reused CRL dir {crl_dir} contains no checkpoints from {CKPT_NAMES}"
        )

    meta_path = crl_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    # Fall back to experiment_summary.json (from run_experiments.py).
    summary_path = crl_dir / "experiment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        overrides = summary.get("overrides", {})
        cfg = CRLConfig()
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return {
            "config":  asdict(cfg),
            "sensors": ["audio", "seismic"],  # run_experiments.py default
            "source":  "experiment_summary.json",
            "overrides": overrides,
        }

    raise FileNotFoundError(
        f"Reused CRL dir {crl_dir} has neither meta.json nor experiment_summary.json"
    )


# ---------------------------------------------------------------------------
# Phase 2 — downstream probes
# ---------------------------------------------------------------------------

def phase_probes(
    cfg: CRLConfig,
    train_ds: SensorDataset,
    val_ds:   SensorDataset,
    device:   torch.device,
    sensors:  list[str],
    crl_dir:  Path,
    probes_root: Path,
    ds_epochs: int,
    pres_weight: torch.Tensor,
    type_weights: torch.Tensor,
    skip_existing: bool,
) -> list[dict]:
    """Run downstream probes (probe_modes × 2 ckpts). Returns per-run summaries."""
    n_probes = len(_probe_modes_for(cfg)) * len(CKPT_NAMES)
    print(f"\n{'=' * 72}\n  PHASE 2 — downstream probes ({n_probes} runs)\n{'=' * 72}")

    single_train = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_single,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    single_val = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_single,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    summaries: list[dict] = []
    probe_modes_active = _probe_modes_for(cfg)
    for probe_mode in probe_modes_active:
        for ckpt_name in CKPT_NAMES:
            run_name = f"{probe_mode}__{Path(ckpt_name).stem}"
            out_dir  = probes_root / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            src_ckpt = crl_dir / ckpt_name
            if not src_ckpt.exists():
                print(f"  [skip] {run_name}: {ckpt_name} missing in CRL dir")
                continue
            done_marker = out_dir / "downstream_best.pth"
            metrics_csv = out_dir / "downstream_metrics.csv"
            meta_path   = out_dir / "meta.json"
            if skip_existing and done_marker.exists() and metrics_csv.exists():
                print(f"  [skip] {run_name}: already complete")
                summaries.append(_probe_summary(run_name, probe_mode, ckpt_name, out_dir))
                continue

            # Mirror the CRL checkpoint so train_downstream finds it at save_dir/<ckpt_name>.
            dst_ckpt = out_dir / ckpt_name
            if not dst_ckpt.exists() or src_ckpt.stat().st_mtime > dst_ckpt.stat().st_mtime:
                shutil.copy2(src_ckpt, dst_ckpt)

            print(f"\n  ── probe: {run_name}")
            # Write meta.json up front so phase_evals can read probe_mode even if
            # training is interrupted mid-run.
            meta_path.write_text(json.dumps({
                "config":     asdict(cfg),
                "sensors":    sensors,
                "probe_mode": probe_mode,
                "ckpt_name":  ckpt_name,
                "crl_run_dir": str(crl_dir.resolve()),
            }, indent=2))

            t0 = time.time()
            model   = CRLModel(cfg, sensors=sensors, probe_mode=probe_mode).to(device)
            trainer = Trainer(model, cfg, device, out_dir)
            trainer.train_downstream(
                single_train, single_val,
                epochs=ds_epochs,
                pres_pos_weight=pres_weight.to(device),
                type_class_weights=type_weights.to(device),
                finetune_top_n=0,
                ckpt_name=ckpt_name,
            )
            elapsed_min = (time.time() - t0) / 60
            # Re-write meta.json to record elapsed time.
            meta = json.loads(meta_path.read_text())
            meta["ds_elapsed_min"] = round(elapsed_min, 2)
            meta_path.write_text(json.dumps(meta, indent=2))
            summaries.append(_probe_summary(run_name, probe_mode, ckpt_name, out_dir))

    return summaries


def _probe_summary(run_name: str, probe_mode: str, ckpt_name: str, out_dir: Path) -> dict:
    """Extract best-epoch metrics from downstream_metrics.csv."""
    import csv as _csv
    csv_path = out_dir / "downstream_metrics.csv"
    best = {
        "val_pres_f1": 0.0, "val_pres_acc": 0.0,
        "val_type_f1": 0.0, "val_type_acc": 0.0,
        "val_loss": float("inf"), "best_epoch": -1,
    }
    if csv_path.exists():
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                vl = float(row.get("val_loss", "inf"))
                if vl < best["val_loss"]:
                    best["val_loss"]     = vl
                    best["best_epoch"]   = int(row["epoch"])
                    best["val_pres_f1"]  = float(row.get("val_pres_f1",  0))
                    best["val_pres_acc"] = float(row.get("val_pres_acc", 0))
                    best["val_type_f1"]  = float(row.get("val_type_f1",  0))
                    best["val_type_acc"] = float(row.get("val_type_acc", 0))
    return {
        "run_name":   run_name,
        "probe_mode": probe_mode,
        "ckpt_name":  ckpt_name,
        "save_dir":   str(out_dir),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
    }


# ---------------------------------------------------------------------------
# Phase 3 — test evals (full × ID × OOD for every probe)
# ---------------------------------------------------------------------------

def phase_evals(
    cfg: CRLConfig,
    test_ds: SensorDataset,
    device:  torch.device,
    sensors: list[str],
    probes_root: Path,
    evals_root:  Path,
    eval_batch_size: int,
    num_workers: int,
    skip_existing: bool,
    recalibrate: bool = False,
) -> list[dict]:
    """Evaluate each downstream model on {full, focal, iobt} splits.

    Shares one preloaded test_ds across all evals by snapshotting then filtering
    test_ds._index in place per run.
    """
    print(f"\n{'=' * 72}\n  PHASE 3 — test evals "
          f"(probes × {len(EVAL_SPLITS)} splits)\n{'=' * 72}")
    orig_index = list(test_ds._index)
    results: list[dict] = []

    probe_dirs = sorted(probes_root.iterdir()) if probes_root.exists() else []
    for probe_dir in probe_dirs:
        if not (probe_dir / "downstream_best.pth").exists():
            continue
        meta_path = probe_dir / "meta.json"
        if not meta_path.exists():
            print(f"  [warn] {probe_dir.name}: meta.json missing, skipping evals")
            continue
        run_name = probe_dir.name
        meta = json.loads(meta_path.read_text())
        probe_mode = meta.get("probe_mode", "linear_ztype")

        # Load the downstream-trained model once, reuse across splits.
        model = CRLModel(cfg, sensors=sensors, probe_mode=probe_mode).to(device)
        model.load_state_dict(torch.load(
            probe_dir / "downstream_best.pth",
            map_location=device, weights_only=True,
        ))
        model.eval()

        for split_name, include in EVAL_SPLITS:
            out_dir = evals_root / run_name / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "eval_report.json"
            if skip_existing and report_path.exists():
                print(f"  [skip] {run_name}/{split_name}")
                results.append(json.loads(report_path.read_text()))
                continue

            # Filter test set in place for this split.
            if include is None:
                test_ds._index = list(orig_index)
            else:
                allowed = set(include)
                test_ds._index = [row for row in orig_index if row[0][0] in allowed]
            if not test_ds._index:
                print(f"  [warn] {run_name}/{split_name}: no windows after filter, skipping")
                continue

            loader = DataLoader(
                test_ds, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_single,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

            print(f"  ── eval: {run_name}/{split_name} "
                  f"({len(test_ds):,} windows)")
            outputs = run_inference(model, loader, device, cfg)
            pres_m = binary_metrics(outputs["pres_logits"], outputs["pres_labels"])
            type_m = (
                multiclass_metrics(outputs["type_logits"], outputs["type_labels"],
                                   N_TYPE_CLASSES)
                if outputs["type_logits"] is not None
                and outputs["type_logits"].numel() > 0
                else None
            )

            report = {
                "run_name":   run_name,
                "probe_mode": probe_mode,
                "ckpt_name":  meta.get("ckpt_name"),
                "split":      split_name,
                "include_datasets": include,
                "n_windows":  len(test_ds),
                "presence":   pres_m,
                "type":       type_m,
            }
            if recalibrate:
                report["presence_target_calibrated"] = recalibrated_binary_metrics(
                    outputs["pres_logits"], outputs["pres_labels"]
                )
                if outputs["type_logits"] is not None \
                        and outputs["type_logits"].numel() > 0:
                    report["type_target_calibrated"] = \
                        recalibrated_multiclass_metrics(
                            outputs["type_logits"], outputs["type_labels"],
                            N_TYPE_CLASSES,
                        )
                else:
                    report["type_target_calibrated"] = None
            report_path.write_text(json.dumps(report, indent=2))

            # Confusion plots (per split).
            _plot_binary_confusion(
                tn=pres_m["tn"], fp=pres_m["fp"], fn=pres_m["fn"], tp=pres_m["tp"],
                title=f"Presence — {run_name} / {split_name}",
                out_path=out_dir / "confusion_presence.png",
            )
            if type_m:
                _plot_confusion_matrix(
                    cm=type_m["confusion_matrix"],
                    class_names=[IDX_TO_CLASS[i] for i in range(N_TYPE_CLASSES)],
                    title=f"Vehicle Type — {run_name} / {split_name}",
                    out_path=out_dir / "confusion_type.png",
                )
            results.append(report)

    # Restore original index so subsequent callers see an unfiltered dataset.
    test_ds._index = orig_index
    return results


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def _load_crl_trajectory(crl_dir: Path) -> dict:
    """Read crl_checkpoint_summary.json if present, else scan crl_metrics.csv.

    Returns best_val_ref_elbo (+ epoch), best_val_aux_type_f1 (+ epoch),
    total_epochs, plus whatever the checkpoint_summary already records.
    """
    summary_path = crl_dir / "crl_checkpoint_summary.json"
    out: dict = {}
    if summary_path.exists():
        out.update(json.loads(summary_path.read_text()))

    # Scan the per-epoch CSV to fill in epoch attribution when missing.
    csv_path = crl_dir / "crl_metrics.csv"
    if csv_path.exists():
        import csv as _csv
        best_elbo      = float("inf"); best_elbo_epoch      = -1
        best_aux_type  = -1.0;         best_aux_type_epoch  = -1
        last_epoch     = -1
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                last_epoch = int(row.get("epoch", -1))
                try:
                    elbo = float(row.get("val_ref_elbo", "inf"))
                except ValueError:
                    elbo = float("inf")
                if elbo < best_elbo:
                    best_elbo = elbo
                    best_elbo_epoch = last_epoch
                try:
                    atf = float(row.get("val_aux_type_f1", 0))
                except ValueError:
                    atf = 0.0
                if atf > best_aux_type:
                    best_aux_type = atf
                    best_aux_type_epoch = last_epoch
        out.setdefault("best_ref_elbo",       round(best_elbo, 6))
        out.setdefault("best_ref_elbo_epoch", best_elbo_epoch)
        out.setdefault("best_aux_type_f1",    round(best_aux_type, 4))
        out.setdefault("best_aux_type_epoch", best_aux_type_epoch)
        out["total_epochs"] = last_epoch + 1
    return out


def write_reports(
    out_dir: Path,
    crl_dir: Path,
    crl_meta: dict,
    probe_summaries: list[dict],
    eval_reports:    list[dict],
) -> None:
    """Write report.json + report.md at the run root."""
    crl_trajectory = _load_crl_trajectory(crl_dir)
    report = {
        "crl": {**crl_meta, "trajectory": crl_trajectory, "crl_dir": str(crl_dir)},
        "probes": probe_summaries,
        "evals":  eval_reports,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # ----- Markdown summary -----
    lines: list[str] = []
    lines.append(f"# Full diagnostic report — {out_dir.name}\n")

    lines.append("## CRL pre-training\n")
    cfg = crl_meta.get("config", {})
    if cfg:
        lines.append(f"- frontend: `{cfg.get('frontend_type')}`, "
                     f"d_z={cfg.get('d_z')}, d_model={cfg.get('d_model')}, "
                     f"n_layers={cfg.get('n_layers')}\n")
    if "crl_elapsed_min" in crl_meta:
        lines.append(f"- elapsed: {crl_meta['crl_elapsed_min']} min\n")
    if crl_meta.get("source") == "experiment_summary.json":
        lines.append(f"- reused from run_experiments.py; overrides: "
                     f"`{crl_meta.get('overrides', {})}`\n")
    if crl_trajectory:
        lines.append(f"- total epochs recorded: {crl_trajectory.get('total_epochs', '?')}\n")
        lines.append(
            f"- **best val_ref_elbo:** {crl_trajectory.get('best_ref_elbo', '?')} "
            f"(epoch {crl_trajectory.get('best_ref_elbo_epoch', '?')}) "
            f"→ `crl_best.pth`\n"
        )
        lines.append(
            f"- **best val_aux_type_f1:** {crl_trajectory.get('best_aux_type_f1', '?')} "
            f"(epoch {crl_trajectory.get('best_aux_type_epoch', '?')}) "
            f"→ `crl_best_aux_type.pth`\n"
        )
    lines.append("")

    lines.append("## Phase 2 — probes (best val by val_loss)\n")
    lines.append("| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in probe_summaries:
        lines.append(
            f"| {s['run_name']} | {s['probe_mode']} | {s['ckpt_name']} | "
            f"{s['best_epoch']} | {s['val_pres_f1']:.4f} | {s['val_type_f1']:.4f} | "
            f"{s['val_type_acc']:.4f} |"
        )
    lines.append("")

    lines.append("## Phase 3 — test evals\n")
    lines.append(
        "Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) "
        "exclude some classes entirely, so `macro_f1_support_only` restricts the "
        "average to classes with support > 0 in that split and is the fair "
        "cross-split comparison.\n"
    )
    has_calibrated = any("type_target_calibrated" in r for r in eval_reports)
    if has_calibrated:
        lines.append(
            "| run | split | n_windows | pres_f1 | type_f1 | "
            "type_f1_support_only | pres_f1_cal | type_f1_cal | "
            "type_f1_support_only_cal | type_acc |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
    else:
        lines.append(
            "| run | split | n_windows | pres_f1 | type_macro_f1 | "
            "type_macro_f1_support_only | type_acc |"
        )
        lines.append("|---|---|---|---|---|---|---|")

    for r in eval_reports:
        pres = r.get("presence") or {}
        typ  = r.get("type") or {}
        per  = typ.get("per_class", {}) if typ else {}
        # Prefer the field computed in eval.py; fall back to recomputation for
        # old reports that don't have it.
        if typ and "macro_f1_support_only" in typ:
            macro_support_only = typ["macro_f1_support_only"]
        else:
            f1_present = [v["f1"] for v in per.values() if v.get("support", 0) > 0]
            macro_support_only = (
                sum(f1_present) / len(f1_present) if f1_present else 0.0
            )
        if has_calibrated:
            pres_cal = r.get("presence_target_calibrated") or {}
            typ_cal  = r.get("type_target_calibrated") or {}
            lines.append(
                f"| {r['run_name']} | {r['split']} | {r['n_windows']:,} | "
                f"{pres.get('f1', 0):.4f} | "
                f"{typ.get('macro_f1', 0):.4f} | "
                f"{macro_support_only:.4f} | "
                f"{pres_cal.get('f1', 0):.4f} | "
                f"{typ_cal.get('macro_f1', 0):.4f} | "
                f"{typ_cal.get('macro_f1_support_only', 0):.4f} | "
                f"{typ.get('accuracy', 0):.4f} |"
            )
        else:
            lines.append(
                f"| {r['run_name']} | {r['split']} | {r['n_windows']:,} | "
                f"{pres.get('f1', 0):.4f} | "
                f"{typ.get('macro_f1', 0):.4f} | "
                f"{macro_support_only:.4f} | "
                f"{typ.get('accuracy', 0):.4f} |"
            )
    lines.append("")

    lines.append("## Per-class type F1 on test splits\n")
    classes = [IDX_TO_CLASS[i] for i in range(N_TYPE_CLASSES)]
    header = "| run | split | " + " | ".join(f"{c}_f1" for c in classes) + " |"
    sep    = "|---|---|" + "|".join(["---"] * len(classes)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in eval_reports:
        typ = r.get("type") or {}
        per = typ.get("per_class", {}) if typ else {}
        row_cells = [r["run_name"], r["split"]]
        for c in classes:
            row_cells.append(f"{per.get(c, {}).get('f1', 0):.3f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {name} ({vram:.0f} GB VRAM)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("  Device: Apple Silicon MPS")
        return torch.device("mps")
    print("  Device: CPU")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    t_start = time.time()

    # Output root
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_root = Path(f"saved_crl/oneshot/{ts}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"  Output root: {out_root}")

    device = get_device()
    cache_dir = Path(args.cache_dir)

    # Base CRLConfig (same knobs as train.py)
    base_cfg = CRLConfig(
        frontend_type=args.frontend,
        morlet_use_phase=args.morlet_use_phase,
        morlet_learnable_w0=args.morlet_learnable_w0,
        morlet_learnable_lr_mult=args.morlet_learnable_lr_mult,
        prior_type=args.prior_type,
        training_mode=args.training_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_epochs=args.crl_epochs,
    )

    # Resolve CRL dir: either reuse or a fresh subdir under out_root.
    if args.crl_run_dir:
        crl_dir = Path(args.crl_run_dir)
        crl_meta = ensure_crl_dir_usable(crl_dir)
        # Rebuild cfg from the saved run so downstream/eval match the CRL training.
        saved_cfg = crl_meta.get("config", {})
        cfg = CRLConfig(**{k: v for k, v in saved_cfg.items()
                           if k in CRLConfig.__dataclass_fields__})
        sensors = crl_meta.get("sensors", args.sensors)
        # Preserve runtime-only overrides.
        cfg.num_workers = args.num_workers
        cfg.batch_size  = args.batch_size
        # Split-mismatch guard. Older runs trained via train.py do NOT write
        # use_id_split into meta even when ID-split was used (train.py wires the
        # flag into SensorDataset directly), so a False meta value can be a
        # false negative. Trust --use-id-split and only block the explicit
        # mismatch where the meta affirmatively says True.
        if saved_cfg.get("use_id_split") is True and not args.use_id_split:
            raise ValueError(
                f"--crl-run-dir was trained with use_id_split=True but "
                f"--use-id-split is not set. Phase 2/3 would read the "
                f"file-based split. Pass --use-id-split --id-root <path>."
            )
        print(f"  Reusing CRL run: {crl_dir}")
    else:
        cfg = base_cfg
        sensors = args.sensors
        crl_dir = out_root / "crl"

    # Honour the CLI flag on cfg so downstream meta.json reflects reality.
    if args.use_id_split:
        cfg.use_id_split = True

    # Preload datasets once.
    print("\nPreloading datasets into shared memory …")
    t_load = time.time()
    if args.use_id_split:
        print(f"  --use-id-split set; reading splits from "
              f"DATASET_VEHICLE_MAP under id_root={args.id_root}")
        id_cache_dir = Path("saved_crl/id_cache")
        train_ds = SensorDataset(
            args.data_dir, cfg, is_train=True, cache_dir=cache_dir,
            use_id_split=True, role="train",
            id_root=args.id_root, id_cache_dir=id_cache_dir,
        )
        val_ds = SensorDataset(
            args.val_dir, cfg, is_train=False, cache_dir=cache_dir,
            use_id_split=True, role="val",
            id_root=args.id_root, id_cache_dir=id_cache_dir,
        )
        test_ds = SensorDataset(
            args.test_dir, cfg, is_train=False, cache_dir=cache_dir,
            use_id_split=True, role="test",
            id_root=args.id_root, id_cache_dir=id_cache_dir,
        )
    else:
        train_ds = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
        val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)
        test_ds  = SensorDataset(args.test_dir, cfg, is_train=False, cache_dir=cache_dir)
    print(f"  Done in {(time.time()-t_load)/60:.1f} min  "
          f"({len(train_ds):,} train / {len(val_ds):,} val / "
          f"{len(test_ds):,} test windows)")

    pres_weight, type_weights = compute_class_weights(train_ds)
    print(f"  Class weights — pres pos_weight: {pres_weight:.3f} | "
          f"type: {[round(w, 3) for w in type_weights.tolist()]}")

    # Phase 1 — CRL
    if not args.crl_run_dir:
        phase_crl(
            cfg=cfg, train_ds=train_ds, val_ds=val_ds,
            device=device, sensors=sensors, crl_dir=crl_dir,
            crl_epochs=args.crl_epochs,
            steps_per_epoch=args.steps_per_epoch,
            skip_existing=args.skip_existing,
        )
        crl_meta = json.loads((crl_dir / "meta.json").read_text())

    # Phase 2 — probes
    probe_summaries = phase_probes(
        cfg=cfg, train_ds=train_ds, val_ds=val_ds,
        device=device, sensors=sensors,
        crl_dir=crl_dir, probes_root=out_root / "downstream",
        ds_epochs=args.ds_epochs,
        pres_weight=pres_weight, type_weights=type_weights,
        skip_existing=args.skip_existing,
    )

    # Phase 3 — evals
    eval_reports = phase_evals(
        cfg=cfg, test_ds=test_ds, device=device, sensors=sensors,
        probes_root=out_root / "downstream",
        evals_root=out_root / "eval",
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
        recalibrate=args.recalibrate,
    )

    # Consolidated report
    write_reports(out_root, crl_dir, crl_meta, probe_summaries, eval_reports)
    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 72}")
    print(f"  Done in {total_min:.1f} min. See {out_root}/report.md")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
