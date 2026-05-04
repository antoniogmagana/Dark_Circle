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
| `morlet_per_sensor` | `_init_morlet_per_sensor` | `sensors` |
| `morlet_fused` | `_init_morlet_fused` | `["fused"]` |
| `morlet_learnable` | `_init_morlet_learnable` | `sensors` |
| `morlet_learnable_fused` | `_init_morlet_learnable_fused` | `["fused"]` |

The legacy SR-heuristic `morlet` variant is removed — `CRLConfig(frontend_type='morlet')` raises with a migration error pointing at `morlet_per_sensor`.

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

### Causal structure (vae mode)

Every forward yields `z` from the encoder, then `model.latent.split(z)` partitions it into:
- `z_pres` (D_PRES=4) → presence head + aux presence head
- `z_type` (D_TYPE=6) → type head + aux type head
- `z_prox` (D_PROX=3) → proximity head
- `z_env` (D_ENV=6) → `UnknownInterventionClassifier` (gated on `cfg.use_interv_classifier`, default off)
- `z_free` (d_z − 19) → unstructured nuisance, no supervision

Aux heads (presence + type) read **μ** rather than the reparameterized sample `z`. Sampling at training time injects noise that hurts partition-routing supervision; the change keeps recon and KL on `z` (correct VAE semantics) while letting aux losses see a deterministic signal. The intervention classifier still takes `z` (sampled) for KL semantics.

Disentangled mode uses a different latent: `SplitLatentSpace(d_z, d_signal=12)` partitions z into `z_signal` and `z_env` only, no per-feature sub-slicing. Aux heads on the mode (sized to `d_signal`) replace the model's D_PRES/D_TYPE-sized heads.

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
trainer.train_crl(
    train_loader, val_loader, epochs,
    pres_pos_weight=...,         # forwarded to mode.set_class_weights
    type_class_weights=...,      # forwarded to mode.set_class_weights
)

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

`mode.set_class_weights(pres_pos_weight, type_class_weights)` runs before the loop. VAE and disentangled modes apply the weights to their aux pres/type losses during CRL — without this, the backbone learns an unweighted representation while the downstream probe applies class weighting, locking a class-imbalanced backbone in place. Contrastive mode ignores the call (no aux losses).

At training end: saves `crl_final.pth` and writes `crl_checkpoint_summary.json` (adding `learned_morlet_params` block for learnable variants).

Per-batch metric `.item()` calls have been hoisted out of the inner loop. Running totals stay as on-device 0-d tensors and are converted to Python floats once per epoch in `_finalize_epoch_metrics`.

### Downstream training loop (`train_downstream`)

1. Load a CRL checkpoint (defaults to `crl_best.pth`; user can pass `crl_best_aux_type.pth`).
2. Freeze or partially unfreeze the backbone per `finetune_top_n` (`0` = fully frozen heads-only, `-1` = full fine-tune, `N ≥ 1` = top N transformer layers + mu/lv heads at 0.1× LR).
3. Build **two independent AdamW optimizers** — one over `pres_heads.parameters()`, one over `type_heads.parameters()`. When backbone is partially unfrozen, the backbone params join both optimizers at 0.1× LR. Each optimizer steps only on its own task's loss, so the per-head Adam moment estimates and LR schedule never couple.
4. Per training batch: forward once, compute `pres_loss = BCE(pres_logits, det)` and `type_loss = CE(type_logits, vtype)` separately, `pres_loss.backward()` then `type_loss.backward()`, then `pres_opt.step()` and `type_opt.step()`. (When backbone is shared between optimizers, the first backward retains the autograd graph so the second can backprop through the shared subgraph too.)
5. Save **two checkpoints** per probe — `downstream_best_pres.pth` at `argmax(val_pres_f1)` and `downstream_best_type.pth` at `argmax(val_type_f1)`. The two heads' best epochs are usually different. Phase 3 evaluates each ckpt against its own task only — never reports a presence number from the type ckpt or vice versa.

The class weighting (`pres_pos_weight`, `type_class_weights`) is computed once from the training set via `compute_class_weights(train_ds)` — uniform effective prior after reweighting. This is why the `probe/recalibration.py` log-prior shift assumes `p_train = uniform`.

When `cfg.use_focal_type=True`, the type head's loss switches to `focal_cross_entropy(weight=type_class_weights, gamma=cfg.focal_type_gamma)` — the focal `(1-p_t)^γ` modulator stacks on top of the inverse-frequency weights, so easy minority-class samples are still up-weighted but hard samples get extra emphasis. Presence BCE is unaffected. The same flag also routes pretraining `aux_type` through focal CE (without class weights — pretraining never used them) so the encoder representation gets the same incentive.

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
