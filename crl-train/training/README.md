# `training/` — model assembly + training loop

Exactly one file that owns:
- **Model assembly**: `CRLModel` builds the network from a `CRLConfig`, dispatching to per-frontend init methods.
- **Training loop**: `Trainer` runs CRL pretraining and downstream probe training, with stage-2 support for two-stage learnable-Morlet workflows.

## File

| File | Contents |
|---|---|
| `trainer.py` | `CRLModel`, `Trainer`, plus metric helpers (`_binary_f1_acc`, `_macro_f1_acc`) |

## `CRLModel`

The `nn.Module` that holds everything: frontends, encoders, decoders, latent space, intervention classifier, probe heads, and aux heads.

### Construction

```python
model = CRLModel(
    config: CRLConfig,
    sensors: list[str] = ["audio", "seismic"],
    probe_mode: str = "linear_ztype",
)
```

Dispatches on `config.frontend_type` in an if/elif ladder:

| frontend_type | Init method | head_keys |
|---|---|---|
| `multiscale` | `_init_multiscale` | `["fused"]` (early fusion) |
| `morlet` | `_init_morlet` | `sensors` (late fusion) |
| `morlet_per_sensor` | `_init_morlet_per_sensor` | `sensors` |
| `morlet_fused` | `_init_morlet_fused` | `["fused"]` |
| `morlet_learnable` | `_init_morlet_learnable` | `sensors` |
| `morlet_learnable_fused` | `_init_morlet_learnable_fused` | `["fused"]` |

Each init method:
1. Builds per-sensor frontends (`MultiScale1DFrontend`, `MorletFilterbank`, or `LearnableMorletFilterbank` wrapped in an `nn.Sequential` with appropriate pooling).
2. For fused topologies: appends `AdaptiveAvgPool1d(T=fused_seq_len)` and builds **one shared** `TemporalEncoder` + `FeatureDecoder`.
3. For per-sensor topologies: builds **one encoder and decoder per sensor**.
4. Records per-sensor `_morlet_derived_params` (kernel_size, pool_stride, target_tokens, etc.) for meta.json audit trail.

### Key public methods

- `encode_fused(x_audio, x_seismic) → (features, z, mu, logvar)` — early-fusion path
- `encode(sensor, x) → (features, z, mu, logvar)` — per-sensor path
- `decode_fused(z) → (B, C, n_sensors*T)`
- `decode(sensor, z) → (B, C, T)`
- `is_fused_frontend() → bool` — used by training modes for dispatch
- `learnable_morlet_parameters() → list[nn.Parameter]` — filter params for the LR group
- `load_from_fixed_morlet_checkpoint(state_dict, strict=True)` — stage-2 conversion (drops kernel_re/im buffers, injects `log_scales`, strips probe heads)
- `backbone_parameters() / head_parameters() / _finetune_params(top_n)` — parameter subsets for optimizer groups

### Causal structure

Every forward yields `z` from the encoder, then `model.latent.split(z)` partitions it into:
- `z_pres` (D_PRES=4) → presence head + aux presence head
- `z_type` (D_TYPE=6) → type head + aux type head
- `z_prox` (D_PROX=3) → proximity head
- `z_env` (D_ENV=6) → `UnknownInterventionClassifier`
- `z_free` (d_z − 19) → unstructured nuisance, no supervision

## `Trainer`

Owns the epoch loop, CSV logging, and patience counting. Algorithm-specific logic lives in `self.mode: TrainingMode`.

### Construction

```python
trainer = Trainer(
    model: CRLModel,
    config: CRLConfig,
    device: torch.device,
    save_dir: Path,
    stage2: bool = False,
)
```

### Optimizer groups

Up to three groups, each with its own LR:

1. **Backbone** — `model.backbone_parameters()`, LR = `config.lr` (or `config.lr × stage2_encoder_lr_mult` when `stage2=True`). Weight decay applied.
2. **Mode params** — learnable sub-modules carried by `self.mode` (e.g., `ConditionalPrior` MLP, contrastive projection head). Same LR as backbone.
3. **Learnable Morlet** (only when applicable) — `model.learnable_morlet_parameters()`, LR = `config.lr × config.morlet_learnable_lr_mult` (default 0.1×). Explicit `name="learnable_morlet"` tag on the group for detection by the scheduler.

`backbone_parameters()` explicitly excludes learnable Morlet params so they don't get double-optimized across groups.

### Scheduler

Two modes:

- **Stage 1 (default)**: plain `CosineAnnealingLR(T_max=config.n_epochs, eta_min=config.lr_min)` on all groups.
- **Stage 2**: `LambdaLR` that implements:
  - Filter group: linear warmup 0 → 1 over first 3 epochs, then cosine annealing over remaining epochs.
  - Backbone + mode groups: cosine annealing from their (already-reduced) peak LR over the full schedule.

Warmup protects the converged encoder from sudden filter drift when stage-2 starts.

### CRL training loop (`train_crl`)

```python
for epoch in range(epochs):
    train_m = self._train_epoch(train_loader, self.beta, steps_per_epoch)
    val_m   = self._eval_epoch(val_loader, self.beta)
    self.scheduler.step()

    new_beta, event = self.mode.update_beta(self.beta, val_m, state, config)
    self.beta = new_beta

    # Log metrics row to crl_metrics.csv
    # Log learnable morlet frequencies (no-op for non-learnable variants)
    # Save checkpoints per mode.should_save_checkpoint()
    # Break on early-stop patience
```

At training end: saves `crl_final.pth` and writes `crl_checkpoint_summary.json` (adding `learned_morlet_params` block for learnable variants).

### Downstream training loop (`train_downstream`)

1. Load a CRL checkpoint (defaults to `crl_best.pth`; user can pass `crl_best_aux_type.pth`).
2. Freeze or partially unfreeze the backbone per `finetune_top_n` (`0` = fully frozen heads-only, `-1` = full fine-tune, `N ≥ 1` = top N transformer layers + mu/lv heads at 0.1× LR).
3. Train `pres_heads`, `type_heads`, `prox_heads` with class-weighted losses for `ds_epochs`.
4. Save `downstream_best.pth` by min `val_loss`.

The class weighting (`pres_pos_weight`, `type_class_weights`) is computed once from the training set via `compute_class_weights(train_ds)` — uniform effective prior after reweighting. This is why the `probe/recalibration.py` log-prior shift assumes `p_train = uniform`.

### Two-stage mechanics

`stage2=True` is signaled only by the Trainer constructor argument. The **conversion logic itself** lives in:

- `CRLModel.load_from_fixed_morlet_checkpoint` (this file) — converts a fixed-Morlet state_dict into the learnable model.
- `crl_vehicle.stage2.find_compatible_run` — auto-finds a compatible source run.
- `crl_vehicle.stage2.resolve_source_checkpoint` — picks `crl_best.pth` (preferred) or falls back to `crl_final.pth`.

`train.py` orchestrates: resolves the source run via `find_compatible_run(auto)` or user-supplied path, loads + converts, then constructs `Trainer(..., stage2=True)` which installs the stage-2 LR groups and scheduler.

## Metric helpers

- `_binary_f1_acc(logits, labels) → (f1, acc)` — presence metric
- `_macro_f1_acc(logits, labels, n_classes) → (f1, acc)` — type metric

Used inside the train/eval epoch loops to aggregate aux head performance. Uses tensor ops only (no sklearn) so it's fast and picklable.
