# CRL-Train

Causal Representation Learning for multi-modal vehicle detection. A modular VAE/contrastive framework with a pluggable frontend, pluggable prior, and pluggable training mode. CITRIS-style temporal intervention matching replaces earlier NOTEARS-style causal discovery.

---

## What this repo does

Trains a latent representation from paired audio (16 kHz) and seismic (100 Hz, post-resample) sensor windows. The representation has a **causal latent structure**: dedicated blocks for presence, vehicle type, proximity, and environment, plus a free noise block. Downstream probe heads (linear, MLP, or full-latent) are trained on the frozen representation to predict presence + vehicle type.

```
raw audio  ┐                                                ┌→ presence
           ├→ Morlet / Multiscale frontend → Transformer ───┤→ vehicle type (4 classes)
raw seismic┘          (fused or per-sensor)   encoder       ├→ proximity
                                              (VAE,         └→ (free nuisance)
                                               disentangled,
                                               or contrastive)
```

The model is **not hardcoded to a specific frontend or training objective**. Five frontends and three training modes are registered; any valid combination runs through the same `Trainer`.

---

## Architecture

### Five frontend variants

| Frontend | Fusion | Kernels | Per-sensor params | Notes |
|---|---|---|---|---|
| `multiscale` | Early | Learnable Conv1D, 3 kernel sizes | Shared `d_model` | Fastest, strongest aggregate F1 today |
| `morlet_per_sensor` | Late | Fixed Morlet, per-sensor freq bands | `morlet_per_sensor_params` | Coupled kernel/stride derivation |
| `morlet_fused` | Early | Fixed Morlet, per-sensor freq bands | Same as above | Morlet kernels + multiscale topology |
| `morlet_learnable` | Late | Learnable log-space scales | Same as above | Stage-2 init optional |
| `morlet_learnable_fused` | Early | Learnable scales, early fusion | Same as above | Combines learnable + fusion |

The legacy SR-heuristic `morlet` variant has been **removed** — its global `morlet_pool_stride=64` collapsed seismic to a single post-pool token at SR=100. `CRLConfig(frontend_type='morlet')` now raises a migration error pointing at `morlet_per_sensor`.

All Morlet variants use **FFT-based convolution** when kernel_size ≥ 512 (automatic; ~3× faster on CPU for audio kernel size 4585). For the fixed-Morlet variants the kernel rFFT is cached per signal length, so each forward pays one input rFFT instead of three.

### Three training modes

- **`vae`** — classical CRL. ELBO + adaptive β annealing + optional intervention classifier + aux heads. Prior is pluggable: `standard` (N(0,I)) or `conditional` (iVAE-style label-conditioned MLP for identifiability). Aux heads read the **deterministic μ** (not the reparameterized sample) and apply the same class weights the downstream probe uses, so the CRL backbone learns a class-balanced representation.
- **`disentangled`** — ELBO over a 2-block (signal/env) `SplitLatentSpace` with cross-modal alignment, env temporal stability, and signal intervention-invariance losses. Aux heads read the full d_signal block (no D_PRES/D_TYPE sub-slicing); same dual-checkpoint shape as `vae`.
- **`contrastive`** — NT-Xent over `StratifiedPairDataset` partners. Same-type and consecutive-window partners are positives; different-type and cross-dataset are negatives. No decoder / KL / aux during CRL; downstream probes train post-hoc on the frozen encoder.

#### Intervention classifier (vae mode only)

The CITRIS-style `UnknownInterventionClassifier` reads the env block but is supervised against `[pres_changed, type_changed]` — a known misrouting that pushes presence/type information INTO the env block. It is **off by default**: `cfg.use_interv_classifier=False`. Set it to `True` only for A/B comparisons against the legacy targeting; `lambda_interv` still controls magnitude when enabled. Disentangled mode replaces this with an explicit invariance loss and ignores the flag.

### Two-stage training

`--init-from-run PATH|auto` loads a converged fixed-Morlet checkpoint (`morlet_per_sensor` or `morlet_fused`), upgrades it to the learnable variant, and fine-tunes. Encoder runs at `lr × stage2_encoder_lr_mult` (default 0.3×) while filters are the primary mover at `lr × morlet_learnable_lr_mult` (default 0.1×). 3-epoch linear warmup on filter LR to avoid shocking the trained encoder.

Rationale: learnable scales trained from scratch chase a moving target (the encoder is drifting too). Two-stage separates "learn a representation" from "tune the filterbank" so the filters see a stable target.

### Performance notes

- **Eval/val DataLoaders cap workers at 8.** Train-side `cfg.num_workers` (often 24) is correct for the partner-sampling + intervention-augmenting pipeline; val and eval don't need that many. `crl_vehicle.seeding.eval_num_workers(cfg.num_workers)` is threaded through every val/eval DataLoader site.
- **Fixed-Morlet kernel rFFT cached** per signal length. The old path recomputed the kernel rFFT every forward; now it's computed once and re-used.
- **Active-only intervention dispatch.** `apply_intervention_batch` only runs the noise generators that appear in the batch's `interv_ids` (and only on the masked subset of rows), instead of running all 7 generators on the full batch and masking after.
- **Per-batch metric `.item()` syncs hoisted out of inner loops** in `Trainer._train_epoch` / `_eval_epoch` and `train_downstream`. Running totals stay on-device until epoch end.

### Sweep tooling

`run_experiments.py --sweep configs/sweeps/<file>.yaml` runs a YAML-driven sweep. Each run spawns `train.py` as a subprocess (GPU + crash isolation); results aggregate into `summary.csv` + `summary.json`. See `configs/sweeps/frontend_comparison.yaml` for the reference sweep over all 5 frontends + phase + learnable_w0.

### Analysis scripts

Post-hoc comparison of completed runs:

- `compare_runs.py` — leaderboard CSV + markdown
- `compare_ablations.py` — pairwise deltas along predefined axes
- `compare_cross_location.py` — per-dataset heatmap + ship metric (min-F1)
- `plot_run.py` — per-run diagnostic plots (training curves, beta, Morlet freq drift, downstream)
- `plot_aggregate.py` — cross-run overlay + bar + scatter
- `gather_top_confusions.py` — re-render confusion matrices for top-ranked runs (uses the same selection rule as `--promote-default`)
- `per_vehicle_confusion.py` — per-(dataset, vehicle, rs_node) confusion breakdown for a single downstream probe; flags reclassification candidates
- `supervised_baseline.py` — end-to-end supervised classifier on the CRL frontend+encoder (no VAE/CRL machinery), as a ceiling for what the data + architecture can achieve

All read recursively from `saved_crl/runs/` and write to `saved_crl/analysis/` by default.

---

## Directory layout

```
crl-train/
├── crl_vehicle/                 # core library (see crl_vehicle/README.md)
│   ├── config.py                # CRLConfig — single source of truth
│   ├── analysis.py              # post-hoc run readers
│   ├── stage2.py                # two-stage helpers: find_compatible_run + conversion
│   ├── data/                    # SensorDataset + StratifiedPairDataset
│   ├── losses/                  # reconstruction, KL, intervention, NT-Xent
│   ├── models/                  # frontends, encoder/decoder, latent split, heads
│   ├── priors/                  # Prior ABC, Standard, Conditional
│   ├── probe/                   # log-prior recalibration
│   └── training_modes/          # TrainingMode ABC, VAE, Disentangled, Contrastive, factory
├── training/
│   └── trainer.py               # CRLModel + Trainer (epoch loop, opt groups, stage-2)
├── configs/
│   └── sweeps/                  # YAML sweep specs
├── tests/                       # 518 tests mirror the module tree
├── saved_crl/                   # run outputs (see analysis scripts)
│   ├── runs/                    # canonical: <frontend>/<training_mode>/<run-id>/{crl,downstream,eval}/
│   │   ├── multiscale/
│   │   ├── morlet_per_sensor/
│   │   ├── morlet_fused/
│   │   ├── morlet_learnable/
│   │   ├── supervised/          # non-CRL baselines: id_split/, file_split/
│   │   └── _archive/            # archived sweeps
│   ├── caches/
│   │   ├── waveform/            # SensorDataset feature cache
│   │   └── id_split/            # ID-split manifest cache
│   ├── analysis/                # compare_runs.py / plot_aggregate.py outputs
│   └── slides-figs/             # session figures, performance tables, notes
│
├── train.py                     # single-run CLI
├── run_full_diagnostic.py       # CRL + downstream + cross-dataset eval in one shot
├── run_experiments.py           # hardcoded list + YAML-driven sweeps
├── eval.py                      # standalone eval of a CRL run on a dataset split
├── validate_run.py              # smoke validation of a completed run
├── smoke_test.py                # tiny synthetic-batch sanity check
│
├── compare_runs.py              # leaderboard
├── compare_ablations.py         # axis deltas
├── compare_cross_location.py    # per-dataset heatmap
├── plot_run.py                  # per-run plots
├── plot_aggregate.py            # cross-run plots
├── gather_top_confusions.py     # confusion matrices for top-ranked runs
├── per_vehicle_confusion.py     # per-(dataset, vehicle) confusion breakdown
├── supervised_baseline.py       # supervised ceiling baseline (no CRL machinery)
```

---

## Quick start

```bash
# Full pipeline: CRL + downstream probes + cross-dataset eval.
python run_full_diagnostic.py --frontend multiscale --crl-epochs 100

# Single frontend, VAE + standard prior, 100 epochs.
python train.py --frontend morlet_per_sensor --crl-epochs 100 \
    --config-overrides-json '{"morlet_use_phase": true}'

# Disentangled mode (replaces the misrouted intervention classifier with an
# explicit signal-invariance loss; aux heads read the full d_signal block).
python train.py --frontend multiscale --training-mode disentangled --crl-epochs 100

# Contrastive NT-Xent, multiscale backbone.
python train.py --frontend multiscale --training-mode contrastive --crl-epochs 100

# Two-stage learnable Morlet (auto-finds the most recent compatible fixed-Morlet run).
python train.py --frontend morlet_learnable \
    --init-from-run auto --crl-epochs 50 \
    --config-overrides-json '{"morlet_use_phase": true}'

# YAML-driven sweep (subprocess per run; GPU/crash isolation).
python run_experiments.py --sweep configs/sweeps/frontend_comparison.yaml

# Post-hoc analysis (run any time after runs finish).
python compare_runs.py
python compare_cross_location.py
python plot_aggregate.py
```

---

## Configuration

`crl_vehicle/config.py::CRLConfig` is the single source of truth. Every knob exposed to the model lives here as a dataclass field; CLI flags are thin wrappers. Major fields:

| Field | Default | Role |
|---|---|---|
| `d_z` | 24 | Latent dim; must exceed `D_CAUSAL = 19` to leave a free subspace |
| `d_model` | 64 | Transformer hidden size + frontend output channels |
| `n_layers` / `n_heads` | 2 / 4 | Transformer shape |
| `frontend_type` | `"multiscale"` | See frontend variants above |
| `training_mode` | `"vae"` | `"vae"`, `"disentangled"`, or `"contrastive"` |
| `prior_type` | `"standard"` | `"standard"` or `"conditional"` (iVAE; vae mode only) |
| `use_interv_classifier` | `False` | When `True`, vae mode includes the CITRIS-style intervention classifier. Default off because it reads `z_env` but is supervised against `[pres_changed, type_changed]` — known to push presence/type info into the env block. Set `True` only for A/B comparisons; `lambda_interv` controls magnitude when enabled. |
| `morlet_use_phase` | `False` | Emit `[log_power, cos_phase, sin_phase]` → 3× channels |
| `morlet_learnable_w0` | `False` | Learnable per-filter bandwidth (only for `morlet_learnable*`) |
| `morlet_learnable_lr_mult` | 0.1 | Filter LR = `lr × this` |
| `stage2_encoder_lr_mult` | 0.3 | Encoder LR when loading a stage-1 checkpoint |
| `n_epochs` | 100 | CRL training epochs |
| `batch_size`, `lr`, `wd` | 128, 3e-4, 1e-4 | Standard optimization hyperparams |
| `early_stop_patience` | 25 | Epochs without val improvement before stopping |
| `kl_floor`, `kl_target`, `beta_step` | 0.01, 0.5, 0.02 | Adaptive-β schedule |
| `use_focal_type` | `False` | When `True`, replace type CE with `(1 - p_t)^γ · weighted_CE` in pretraining aux_type and downstream probe. Stacks on existing `type_class_weights`; presence BCE unaffected |
| `focal_type_gamma` | 2.0 | Focal exponent γ. Ignored unless `use_focal_type=True` |

Per-sensor Morlet knobs live in `morlet_per_sensor_params`:

```python
{
    "audio":   {"freq_min": 20.0,  "freq_max": 8000.0, "w0": 6.0,
                "out_channels_frac": 1.0, "target_tokens": 32,
                "receptive_cycles": 3.0},
    "seismic": {"freq_min": 2.0,   "freq_max": 40.0,   "w0": 6.0,
                "out_channels_frac": 1.0, "target_tokens": 32,
                "receptive_cycles": 3.0},
}
```

`kernel_size` and `pool_stride` are **derived** from these — not hardcoded. See `crl_vehicle/models/README.md` for the formulas.

---

## Run outputs

Each run writes to a `save_dir` (default: `saved_crl/runs/<frontend>/<training_mode>/<timestamp>/`):

```
<save_dir>/
├── crl/
│   ├── meta.json                    # config + sensors + stage2 attribution
│   ├── crl_metrics.csv              # per-epoch training curves
│   ├── crl_checkpoint_summary.json  # best ELBO, best aux_type_f1, learned_morlet_params
│   ├── crl_best.pth                 # val_ref_elbo-selected checkpoint
│   ├── crl_best_aux_type.pth        # val_aux_type_f1-selected checkpoint
│   ├── crl_final.pth                # last-epoch checkpoint
│   └── learnable_morlet_freqs.csv   # (only for learnable variants)
├── downstream/<probe_mode>__<ckpt>/
│   ├── downstream_metrics.csv       # per-epoch probe training
│   ├── downstream_best_pres.pth     # argmax val_pres_f1 (deployed to detection node)
│   ├── downstream_best_type.pth     # argmax val_type_f1 (deployed to classification node)
│   └── meta.json
└── eval/<probe_mode>__<ckpt>/<head>/<split>/
    ├── eval_report.json             # only this head's metrics
    └── confusion_*.png              # (head ∈ {pres, type})
```

Splits: `iobt`, `focal`, `m3nvc` (per-dataset), `full` (pooled), plus per-vehicle splits like `focal__pickup2`, `m3nvc__cx30`.
Probe modes: `linear_ztype`, `mlp_ztype`, `linear_fullz`.
CRL checkpoint names: `crl_best.pth` (ELBO), `crl_best_aux_type.pth` (F1 — epoch-invariant, preferred when training uses β annealing).
Probe checkpoint names: `downstream_best_pres.pth` (argmax `val_pres_f1`), `downstream_best_type.pth` (argmax `val_type_f1`). The two heads train jointly with two independent optimizers — the pres optimizer steps only on the BCE presence loss, the type optimizer steps only on the CE type loss, so the heads don't fight each other for shared LR/Adam state.

---

## Deploying a checkpoint to the inference engine

Use `export_for_inference.py` to convert a probe-trained run into the
TorchScript bundles the inference pods consume. The two inference pods
(`infer-detect` and `infer-classify`) are deployed independently and
read from **two separate catalogs**:

- `inference-engine/detect-bundles/` — encoder + presence head, selected at build time by `DETECT_BUNDLE` (default `detect-default`).
- `inference-engine/classify-bundles/` — encoder + type head, selected by `CLASSIFY_BUNDLE` (default `classify-default`).

Each invocation produces **one** bundle; export both kinds from the
same saved run with two calls. The exporter writes directly into the
appropriate catalog when given `--bundle-name`. Assumes `crl-train`
and `inference-engine` are siblings under one parent directory (the
standard project layout); override with `--bundles-dir` otherwise.

```bash
# Detect bundle (encoder + presence head)
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind detect \
    --bundle-name <frontend>-<mode>-<run>-v<N>

# Classify bundle (encoder + type head)
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/<frontend>/<mode>/<run>/downstream/<probe> \
    --bundle-kind classify \
    --bundle-name <frontend>-<mode>-<run>-<probe>-v<N>
```

Naming convention (enforced by the exporter — fails fast if the name
doesn't end in `-v<N>`):
- detect:   `<frontend>-<training-mode>-<run-id>-v<N>`
- classify: `<frontend>-<training-mode>-<run-id>-<probe>-v<N>`

Example for the current shipping leader:

```bash
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/multiscale/vae/v3_lowfreq/downstream/mlp_ztype__crl_best_aux_type \
    --bundle-kind classify \
    --bundle-name multiscale-vae-v3_lowfreq-mlp_ztype-v2
```

To evaluate the new bundle against its catalog and repoint the
`<kind>-default` symlink if it wins, add `--promote-default`:

```bash
poetry run python export_for_inference.py \
    --save-dir saved_crl/runs/multiscale/vae/v3_lowfreq/downstream/mlp_ztype__crl_best_aux_type \
    --bundle-kind classify \
    --bundle-name multiscale-vae-v3_lowfreq-mlp_ztype-v2 \
    --promote-default
```

The selection rule is run by the exporter, not by hand — it enforces
catalog floors so a regression isn't auto-promoted into the shipping
default.

After exporting:

1. Update the catalog table in
   `inference-engine/<kind>-bundles/README.md` with the new bundle's
   pres_f1 / type_f1 / min_*_f1 numbers.
2. Commit the bundle directory and (if you promoted) the symlink change.

For ad-hoc exports outside the bundle catalog (parity testing, scratch
deploys), use `--out-dir <path>` instead of `--bundle-name` — that flag
skips the naming check and writes wherever you point it.

---

## Testing

```bash
python -m pytest tests/ -q
```

518 tests across `tests/` mirroring the module tree. Run pytest against `tests/` directly — the top-level `smoke_test.py` is a script, not a pytest module, and pytest will fail at collection if pointed at the repo root.

---

## Where to read next

Start with the nested READMEs for specifics on each layer:

- **`crl_vehicle/README.md`** — package overview, config dataclass, post-hoc analysis helpers
- **`crl_vehicle/data/README.md`** — dataset format, stratified partner sampling, interventions
- **`crl_vehicle/models/README.md`** — all 5 frontends, encoder/decoder, latent split, heads
- **`crl_vehicle/losses/README.md`** — reconstruction, KL, intervention matching, NT-Xent
- **`crl_vehicle/priors/README.md`** — Prior ABC, StandardPrior, ConditionalPrior (iVAE)
- **`crl_vehicle/probe/README.md`** — log-prior recalibration (target-prior shift)
- **`crl_vehicle/training_modes/README.md`** — TrainingMode ABC, VAE, Contrastive, factory
- **`training/README.md`** — Trainer class, CRLModel assembly, two-stage mechanics
