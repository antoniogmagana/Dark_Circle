# `crl_vehicle/` — core library

Python package that holds everything the training pipeline and analysis scripts depend on. Entry-point scripts (`train.py`, `run_full_diagnostic.py`, `compare_*.py`, `plot_*.py`) import from here.

## Top-level modules

| File | Purpose |
|---|---|
| `config.py` | `CRLConfig` dataclass — single source of truth for every training knob. Also `ModalityConfig`, `DATASET_VEHICLE_MAP`, `CATEGORY_TO_IDX`, `LABEL_BACKGROUND`, `LABEL_MULTI`. |
| `analysis.py` | Post-hoc run readers (`discover_runs`, `load_run_metrics`, `load_crl_timeseries`, `apply_filters`). Used by `compare_*.py` and `plot_*.py`. Read-only; never mutates run dirs. |
| `seeding.py` | Determinism helpers: `seed_everything`, `seeded_dataloader_kwargs(seed)` for spread into `DataLoader(...)`, and `eval_num_workers(train_n)` which caps val/eval/inference DataLoader worker counts at 8 (train-side `cfg.num_workers` is correct for the partner-sampling + augmentation pipeline; val/eval don't need that many). |
| `stage2.py` | Two-stage training helpers. `find_compatible_run` searches for a compatible stage-1 run given a target learnable config; `resolve_source_checkpoint` picks the right `.pth` to load. |

## Subpackages

| Subpackage | One-line summary |
|---|---|
| `data/` | `SensorDataset` + `StratifiedPairDataset` + collate functions + 7 intervention noise types |
| `losses/` | Reconstruction, KL divergence, intervention matching BCE, NT-Xent contrastive |
| `models/` | Frontends (5 variants), Transformer encoder/decoder, latent splits (`CausalLatentSpace`, `SplitLatentSpace`), intervention classifier, downstream heads |
| `priors/` | `Prior` ABC + `StandardPrior` (N(0,I)) + `ConditionalPrior` (iVAE label-conditioned MLP) |
| `probe/` | Log-prior shift for probe evaluation (target-prior-aware classification) |
| `training_modes/` | `TrainingMode` ABC + `VAETrainingMode` + `DisentangledVAETrainingMode` + `ContrastiveTrainingMode` + factory |

Each subpackage has its own README with file-level detail.

## `CRLConfig` (the object that ties everything together)

Dataclass in `config.py`. Passed to:
- `CRLModel(config, sensors, probe_mode)` — builds the network per `config.frontend_type`
- `Trainer(model, config, device, save_dir, stage2=bool)` — owns the optimizer and loop
- `build_training_mode(config)` — returns the right `TrainingMode` per `config.training_mode` + `config.prior_type`
- `SensorDataset(path, config, is_train=..., cache_dir=...)` — uses window size + sample rate from `config.modality_cfg(sensor)`

**Never** import `CRLConfig` fields globally at module scope; pass the `config` instance through. This keeps components data-agnostic so the same model code supports different `d_model` / `window_size` / per-sensor params without touching it.

## Adding a new component

The registry pattern across the codebase is:

1. **Define the class** in its natural subpackage (e.g., a new frontend → `models/frontend.py`).
2. **Add a `CRLConfig` field** or enum value in `config.py` that selects it.
3. **Wire the dispatch** in `training/trainer.py::CRLModel.__init__` (the `if/elif` ladder on `config.frontend_type`).
4. **Update shape mappings** — for frontends, `_init_morlet_*` methods also populate `_morlet_derived_params`; for training modes, extend the factory.
5. **Tests** mirroring the module tree under `tests/`.

See `tests/models/test_frontend.py` for the template — each variant has its own test class.
