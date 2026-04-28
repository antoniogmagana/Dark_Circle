# `crl_vehicle/training_modes/` — training algorithms

Each `TrainingMode` encapsulates one coherent CRL training program (loss, β schedule, checkpoint selection, early-stop metric). `Trainer` is mode-agnostic: it owns the epoch loop, CSV logging, and patience counting, and delegates algorithm-specific decisions to the mode.

## Files

| File | Contents |
|---|---|
| `base.py` | `TrainingMode(nn.Module, ABC)` + `CheckpointState` dataclass |
| `vae_mode.py` | `VAETrainingMode` — ELBO + β schedule + intervention matching + aux heads |
| `contrastive_mode.py` | `ContrastiveTrainingMode` — NT-Xent over stratified partners |
| `disentangled_mode.py` | `DisentangledVAETrainingMode` — ELBO over a 2-block (signal/env) `SplitLatentSpace` with cross-modal alignment, env temporal stability, and signal intervention-invariance losses. Aux heads read the full d_signal block (no D_PRES/D_TYPE sub-slicing). Same dual-checkpoint shape as `VAETrainingMode`. |
| `factory.py` | `build_training_mode(config)` — picks the right mode + prior |
| `__init__.py` | Re-exports: `TrainingMode`, `CheckpointState`, `VAETrainingMode`, `ContrastiveTrainingMode`, `DisentangledVAETrainingMode`, `build_training_mode` |

## The interface

```python
class TrainingMode(nn.Module, ABC):
    @abstractmethod
    def forward_pair(model, batch, beta, device) -> (loss, metrics_dict)
    @abstractmethod
    def val_metrics_summary(val_m: dict) -> dict
    @abstractmethod
    def update_beta(beta, val_m, state, config) -> (new_beta, event_string)
    @abstractmethod
    def should_save_checkpoint(val_m, epoch, state) -> dict[str, bool]
    @abstractmethod
    def early_stop_metric() -> str
    @abstractmethod
    def early_stop_mode() -> str    # "min" or "max"

    def checkpoint_summary(state) -> dict
```

Modes are `nn.Module` subclasses so learnable sub-modules they carry (iVAE's conditional prior, the contrastive projection head) get discovered by `Trainer`'s param-group logic. The Trainer does:

```python
backbone_ids = {id(p) for p in model.backbone_parameters()}
mode_params = [p for p in self.mode.parameters() if id(p) not in backbone_ids]
```

This lifts mode params into their own optimizer group cleanly.

## `VAETrainingMode`

Classical CRL. Implements:

- **Forward dispatch** by frontend topology: `model.is_fused_frontend()` → `_forward_pair_fused` (shared encoder, one head key `"fused"`) vs `_forward_pair_per_sensor` (per-sensor encoders, one head key per sensor).
- **Loss** = `recon + β·kl + λ_interv·interv + λ_aux_pres·aux_pres + λ_aux_type·aux_type`.
  - `aux_type` is `F.cross_entropy(unweighted)` by default, or `focal_cross_entropy(weight=None, gamma=cfg.focal_type_gamma)` when `cfg.use_focal_type=True`. `aux_pres` is BCE and not affected by the focal flag.
- **KL computation** via the injected `Prior`. Y = `stack(det_t, type_t)` plumbed through even to `StandardPrior` (which ignores it).
- **Adaptive β schedule** in `update_beta`:
  - If raw_kl below `config.kl_floor` → β decreases (prior collapsing).
  - If recon still improving OR raw_kl above `config.kl_target` → β increases.
  - Otherwise hold.
- **Dual checkpoint** in `should_save_checkpoint`:
  - `crl_best.pth` — selected by `val_ref_elbo` (= val_recon + val_raw_kl, **β-independent reference ELBO**). Epoch-invariant across the adaptive β schedule.
  - `crl_best_aux_type.pth` — selected by `val_aux_type_f1` (downstream-proxy signal that doesn't depend on β at all).
  - Both fire independently; `patience_count` resets only when `val_ref_elbo` improves (early-stop metric).
- **Early-stop metric**: `val_ref_elbo` with `mode="min"`.

The β-invariant reference ELBO was introduced to fix a subtle issue from before Checkpoint 1: the old `val_elbo` included the current β, so ELBO from early epochs (low β, low KL weight) was incomparable to ELBO from later epochs (high β, high KL weight). Selecting the "best" checkpoint by that metric was hostage to the β schedule. `val_ref_elbo` evaluates at β=1 for checkpoint selection, leaving the training loss schedule free to anneal.

## `ContrastiveTrainingMode`

NT-Xent over stratified partners. Implements:

- **Forward dispatch** by topology (same pattern as VAE): `_encode_fused` vs `_encode_per_sensor`, both producing `mu` (deterministic — no sampling) + L2-normalized projection.
- **Projection head** is a small `d_z → d_proj → d_proj` MLP that lives on the mode, not on `CRLModel`. Keeps the model VAE-shaped; mode params flow through the optimizer's mode-params group.
- **Positives**: `STRATUM_CONSEC` + `STRATUM_SAME_TYPE`. **Negatives**: `STRATUM_DIFF_TYPE` + `STRATUM_CROSS_DS` for this anchor, plus all other anchors' partners in the batch.
- **No decoder / KL / aux / intervention** during CRL. The encoder is trained purely by NT-Xent; downstream probes train post-hoc on the frozen encoder.
- **`update_beta`** returns `(0.0, "→hold")` — contrastive doesn't use β, but the interface requires the method.
- **Single checkpoint** `crl_best.pth` selected by `val_contrastive_loss` (lower = better).
- **Early-stop metric**: `val_contrastive_loss` with `mode="min"`.

## `factory.py`

```python
build_training_mode(config) -> TrainingMode
```

Dispatches on `(config.training_mode, config.prior_type)`:

| training_mode | prior_type | Returns |
|---|---|---|
| `"vae"` | `"standard"` | `VAETrainingMode(prior=StandardPrior(), config)` |
| `"vae"` | `"conditional"` | `VAETrainingMode(prior=ConditionalPrior(d_z), config)` |
| `"contrastive"` | `"standard"` | `ContrastiveTrainingMode(config)` |
| `"contrastive"` | anything else | **raises** (contrastive doesn't use a prior) |

Any unknown combination raises with a list of valid options.

## Adding a new training mode

1. Subclass `TrainingMode` in a new file `training_modes/<name>_mode.py`.
2. Implement all six abstract methods. Forward_pair must dispatch on `model.is_fused_frontend()` if your loss depends on the topology.
3. Add any mode-owned `nn.Module` sub-components (like contrastive's projection) in `__init__` — they'll flow through the optimizer automatically via mode-params.
4. Add a branch in `factory.build_training_mode`.
5. Mirror the test structure from `tests/training_modes/test_vae_mode.py` or `test_contrastive_mode.py`: factory tests, forward-pair dispatch tests, checkpoint-logic tests.
