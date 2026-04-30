"""CRL training entry point."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crl_vehicle.config import CRLConfig
from crl_vehicle.data.dataset import (
    SensorDataset, StratifiedPairDataset,
    collate_pairs, collate_single,
    compute_class_weights,
)
from training.trainer import CRLModel, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CRL model")
    p.add_argument("--phase", choices=["crl", "downstream", "full"], default="full")
    p.add_argument("--data-dir",  default="../data_files/parsed/train/")
    p.add_argument("--val-dir",   default="../data_files/parsed/val/")
    p.add_argument("--use-id-split", action="store_true",
                   help="Use the in-distribution split schema "
                        "(see docs/superpowers/specs/2026-04-25-id-split-schema-design.md). "
                        "When set, train/val/test assignments come from "
                        "DATASET_VEHICLE_MAP markers; --data-dir and --val-dir are ignored.")
    p.add_argument("--id-root", default="../data_files/parsed/",
                   help="Parent dir containing train/, val/, test/. "
                        "Used only when --use-id-split is set.")
    p.add_argument("--sensors",   nargs="+", default=["audio", "seismic"])
    p.add_argument("--crl-epochs",  type=int,   default=100)
    p.add_argument("--ds-epochs",   type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num-workers", type=int,   default=8)
    p.add_argument("--save-dir",    default=None)
    p.add_argument("--frontend",    choices=["multiscale", "morlet", "morlet_per_sensor",
                                              "morlet_fused", "morlet_learnable",
                                              "morlet_learnable_fused"],
                   default="multiscale",
                   help="Legacy frontend selector. Translates to "
                        "(--frontend-bank, --frontend-fusion). Prefer the new "
                        "two-flag form for new experiments.")
    p.add_argument("--frontend-bank",
                   choices=["multiscale", "morlet", "morlet_learnable"],
                   default=None,
                   help="New two-axis frontend selector (bank). When set, "
                        "overrides --frontend translation.")
    p.add_argument("--frontend-fusion",
                   choices=["late", "early"],
                   default=None,
                   help="New two-axis frontend selector (fusion). When set, "
                        "overrides --frontend translation.")
    p.add_argument("--audio-target-rate", type=int, default=None,
                   help="Audio resample target rate. Default 16000. Set lower "
                        "(e.g. 4000) when only the sub-2 kHz vehicle band "
                        "matters; the dataset cache key auto-versions on this.")
    p.add_argument("--morlet-learnable-w0", action="store_true",
                   help="Make per-filter w0 learnable (only applies to "
                        "morlet_learnable / morlet_learnable_fused).")
    p.add_argument("--morlet-learnable-lr-mult", type=float, default=0.1,
                   help="LR multiplier for learnable Morlet params relative to "
                        "backbone LR (default 0.1).")
    p.add_argument("--training-mode", choices=["vae", "contrastive", "disentangled"], default="vae",
                   help="'vae' = ELBO + aux + interv (default). 'contrastive' = "
                        "NT-Xent over stratified partners during CRL.")
    p.add_argument("--use-focal-type", action="store_true",
                   help="Replace type CE with focal CE in pretraining aux_type and "
                        "downstream probe. Stacks on existing class weights.")
    p.add_argument("--focal-type-gamma", type=float, default=2.0,
                   help="Focal CE gamma for the type loss (default 2.0; ignored "
                        "unless --use-focal-type is set).")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Limit batches per epoch (for smoke tests)")
    p.add_argument("--cache-dir",   default="./saved_crl/caches/waveform")
    # Downstream fine-tuning options
    p.add_argument("--finetune-top-n", type=int, default=0,
                   help="Unfreeze top N encoder transformer layers during downstream "
                        "(0 = fully frozen backbone, -1 = unfreeze all)")
    p.add_argument("--probe-mode",
                   choices=["linear_ztype", "mlp_ztype", "linear_fullz",
                            "linear_signal", "mlp_signal"],
                   default="linear_ztype",
                   help="Downstream type-head architecture. linear_ztype: Linear(6,4) on "
                        "z_type (default); mlp_ztype: Linear(6,32)-ReLU-Linear(32,4) on "
                        "z_type; linear_fullz: Linear(d_z,4) on the full latent; "
                        "linear_signal: Linear(d_signal,4) on z[0:d_signal] (disentangled); "
                        "mlp_signal: Linear(d_signal,32)-ReLU-Linear(32,4) on z[0:d_signal] "
                        "(disentangled).")
    p.add_argument("--crl-run-dir", default=None,
                   help="Existing run dir to load crl_best.pth from. When set, --save-dir "
                        "defaults to <crl-run-dir>/probes/<probe-mode>/ so probe artifacts "
                        "don't overwrite the original downstream outputs.")
    p.add_argument("--ckpt-name", default="crl_best.pth",
                   help="Which CRL checkpoint to load for downstream. Options: "
                        "crl_best.pth (val_ref_elbo, default), crl_best_aux_type.pth "
                        "(val_aux_type_f1), crl_final.pth (last epoch).")
    p.add_argument("--config-overrides-json", default=None,
                   help="JSON dict of CRLConfig field overrides applied on top of "
                        "the other CLI args. Used by run_experiments.py --sweep "
                        "to pass arbitrary config fields the other flags don't "
                        "expose. Unknown fields raise.")
    p.add_argument("--init-from-run", default=None, metavar="PATH_OR_AUTO",
                   help="Two-stage training: load the CRL backbone from a "
                        "converged stage-1 run (morlet_per_sensor or "
                        "morlet_fused), upgrade to the requested learnable "
                        "variant, and fine-tune. Pass 'auto' to search "
                        "saved_crl/runs/ for the most recent compatible "
                        "run. Only valid with --frontend morlet_learnable or "
                        "morlet_learnable_fused.")
    p.add_argument("--init-search-root", default="saved_crl/runs",
                   help="Root dir searched by --init-from-run=auto. Default: "
                        "saved_crl/runs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_kwargs = dict(
        frontend_type=args.frontend,
        training_mode=args.training_mode,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        n_epochs=args.crl_epochs,
        morlet_learnable_w0=args.morlet_learnable_w0,
        morlet_learnable_lr_mult=args.morlet_learnable_lr_mult,
        use_focal_type=args.use_focal_type,
        focal_type_gamma=args.focal_type_gamma,
    )
    if args.frontend_bank is not None:
        cfg_kwargs["frontend_bank"] = args.frontend_bank
    if args.frontend_fusion is not None:
        cfg_kwargs["frontend_fusion"] = args.frontend_fusion
    if args.audio_target_rate is not None:
        cfg_kwargs["audio_target_rate"] = args.audio_target_rate
    cfg = CRLConfig(**cfg_kwargs)

    if args.config_overrides_json is not None:
        import json as _json
        overrides = _json.loads(args.config_overrides_json)
        for k, v in overrides.items():
            if k not in CRLConfig.__dataclass_fields__:
                raise ValueError(
                    f"--config-overrides-json: unknown CRLConfig field {k!r}. "
                    f"Valid: {sorted(CRLConfig.__dataclass_fields__.keys())}"
                )
            setattr(cfg, k, v)

    # Mirror the CLI flag onto cfg so the saved meta.json honestly reflects
    # which split this run consumed. Otherwise meta lies and downstream tools
    # (run_full_diagnostic.py --crl-run-dir) can't tell ID-split runs apart
    # from file-split runs.
    cfg.use_id_split = args.use_id_split

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.crl_run_dir is not None:
        crl_run_dir = Path(args.crl_run_dir)
        default_save = crl_run_dir / "probes" / f"{args.probe_mode}__{Path(args.ckpt_name).stem}"
        save_dir = Path(args.save_dir) if args.save_dir else default_save
        save_dir.mkdir(parents=True, exist_ok=True)
        # Mirror the selected CRL checkpoint into save_dir so train_downstream finds it.
        src_ckpt = crl_run_dir / args.ckpt_name
        dst_ckpt = save_dir / args.ckpt_name
        if not src_ckpt.exists():
            raise FileNotFoundError(
                f"--crl-run-dir missing {args.ckpt_name}: {src_ckpt}"
            )
        if not dst_ckpt.exists() or src_ckpt.stat().st_mtime > dst_ckpt.stat().st_mtime:
            shutil.copy2(src_ckpt, dst_ckpt)
    else:
        save_dir = Path(args.save_dir or f"saved_crl/{args.frontend}/{run_ts}")
        save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.use_id_split:
        print(f"INFO: --use-id-split is set; --data-dir and --val-dir are ignored, "
              f"reading splits from DATASET_VEHICLE_MAP under id_root={args.id_root}")
        id_cache_dir = Path("saved_crl/caches/id_split")
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
    else:
        train_ds = SensorDataset(args.data_dir, cfg, is_train=True,  cache_dir=cache_dir)
        val_ds   = SensorDataset(args.val_dir,  cfg, is_train=False, cache_dir=cache_dir)

    pres_weight, type_weights = compute_class_weights(train_ds)
    print(f"  Class weights — pres pos_weight: {pres_weight:.3f} | "
          f"type: {[round(w, 3) for w in type_weights.tolist()]}")

    model   = CRLModel(cfg, sensors=args.sensors, probe_mode=args.probe_mode).to(device)

    # Two-stage training: load a converged fixed-Morlet checkpoint and
    # upgrade to the learnable variant. Only valid for morlet_learnable*.
    stage2 = False
    init_from_run_resolved: Path | None = None
    if args.init_from_run is not None:
        from crl_vehicle.stage2 import find_compatible_run, resolve_source_checkpoint
        if args.frontend not in ("morlet_learnable", "morlet_learnable_fused"):
            raise ValueError(
                f"--init-from-run requires --frontend morlet_learnable or "
                f"morlet_learnable_fused (got {args.frontend!r})."
            )
        if args.init_from_run == "auto":
            print(f"Searching {args.init_search_root} for compatible stage-1 run …")
            init_from_run_resolved = find_compatible_run(
                target_frontend=args.frontend,
                target_sensors=args.sensors,
                target_cfg=asdict(cfg),
                search_root=Path(args.init_search_root),
                verbose=True,
            )
            print(f"Auto-selected stage-1 run: {init_from_run_resolved}")
        else:
            init_from_run_resolved = Path(args.init_from_run)
            if not init_from_run_resolved.exists():
                raise FileNotFoundError(
                    f"--init-from-run path does not exist: {init_from_run_resolved}"
                )

        ckpt_path = resolve_source_checkpoint(init_from_run_resolved)
        print(f"Loading stage-1 checkpoint: {ckpt_path}")
        source_state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_from_fixed_morlet_checkpoint(
            source_state, strict=True,
        )
        print(f"  missing keys (expected for fresh heads / w0): {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
        stage2 = True

    trainer = Trainer(model, cfg, device, save_dir, stage2=stage2)

    if args.phase in ("crl", "full"):
        train_pair = StratifiedPairDataset(train_ds)
        val_pair   = StratifiedPairDataset(val_ds)
        train_loader = DataLoader(
            train_pair, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_pairs, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_pair, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_pairs, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        trainer.train_crl(
            train_loader, val_loader,
            epochs=args.crl_epochs,
            steps_per_epoch=args.steps_per_epoch,
        )

    if args.phase in ("downstream", "full"):
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_single, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_single, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        trainer.train_downstream(
            train_loader, val_loader,
            epochs=args.ds_epochs,
            pres_pos_weight=pres_weight.to(device),
            type_class_weights=type_weights.to(device),
            finetune_top_n=args.finetune_top_n,
            ckpt_name=args.ckpt_name,
        )

    meta: dict = {
        "config":     asdict(cfg),
        "sensors":    args.sensors,
        "probe_mode": args.probe_mode,
        "ckpt_name":  args.ckpt_name,
        "crl_run_dir": str(Path(args.crl_run_dir).resolve()) if args.crl_run_dir else None,
        "stage2":       stage2,
        "init_from_run": str(init_from_run_resolved.resolve())
                         if init_from_run_resolved is not None else None,
    }
    derived = getattr(model, "_morlet_derived_params", None)
    if derived:
        meta["morlet_derived_params"] = derived
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Artifacts in {save_dir}")


if __name__ == "__main__":
    main()
