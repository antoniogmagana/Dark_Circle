# Deferred implementations

Lower-priority items captured for future sessions. Not blocking current work.

## β schedule: warmup + linear ramp (alternative to adaptive)

**Problem.** The current `update_beta` in `vae_mode.py` and `disentangled_mode.py`
uses an adaptive schedule keyed on `val_raw_kl` and `val_recon` deltas. β
moves epoch-by-epoch, so the encoder chases a moving optimum:
posterior shape at β=0.1 differs from posterior shape at β=1.0, encoder never
settles. Symptom in past runs: `raw_kl` blowup (morlet_learnable runs hit
~600+) or downstream F1 peak at early epoch then decay.

**Proposed schedule.** Two-phase fixed:
1. **Warmup**: hold β=0 for N epochs. Encoder converges to an unconstrained
   representation (recon + physics losses do the work).
2. **Ramp**: linearly increase β from 0 → β_target over M epochs.
3. **Hold**: β=β_target for the remainder.

For 100-epoch runs: warmup=20, ramp=30, target=1.0 (or 0.3 for disentangled
mode where physics losses already do identifiability work).

**Why not jump β=0→1.0 directly.** Encoder under β=0 produces high-magnitude
μ and very negative logvar — KL penalty at β=1 spikes to thousands, gradient
collapses or oscillates. Same failure mode as the morlet_learnable runs that
diverged.

**Implementation sketch.**

Config additions:
```python
beta_schedule: str = "adaptive"        # default keeps current behavior
# alternatives: "warmup_ramp"
beta_warmup_epochs: int = 20
beta_ramp_epochs:   int = 30
beta_target:        float = 1.0
```

`CheckpointState` needs an `epoch: int = 0` field (Trainer increments per epoch).

Each TrainingMode's `update_beta` checks `config.beta_schedule`:
- `"adaptive"` → existing logic
- `"warmup_ramp"` → epoch-based:
  ```python
  if epoch < cfg.beta_warmup_epochs:
      return (0.0, "→warmup")
  elif epoch < cfg.beta_warmup_epochs + cfg.beta_ramp_epochs:
      progress = (epoch - cfg.beta_warmup_epochs) / cfg.beta_ramp_epochs
      return (cfg.beta_target * progress, f"↑ramp[{progress:.2f}]")
  else:
      return (cfg.beta_target, "→target")
  ```

**Caveat for checkpoint selection.** `val_ref_elbo = recon + raw_kl @ β=1`
penalizes lower-β-target runs harder. Use `val_aux_type_f1` for selection
when β_target < 1. The dual-checkpoint structure already handles this — just
report `crl_best_aux_type.pth` numbers in the leaderboard.

**Tests needed.**
- Warmup phase: β stays at 0 across N epochs.
- Ramp phase: β interpolates linearly from 0 to target.
- Hold phase: β stays at target.
- Schedule selection via config dispatches to the right path.

**Estimated cost.** ~30 lines (mode update + config + dispatch) + ~6 tests.

**When to revisit.** If disentangled mode v1 shows the same chasing/collapse
pattern as morlet_learnable (raw_kl growing through training, downstream F1
peaking early then decaying), this is the first thing to try.

---

## CUDA-aware test fixture

**Problem.** `test_disentangled_mode.py` runs all tests on CPU. Device-mismatch
bugs (CPU-tensor masking dev-tensor) are invisible to CI. The original
disentangled mode shipped with one such bug, caught only on first server run.

**Proposed.** Add a parametrized fixture that runs key forward-pass tests on
CUDA when available:
```python
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
def device(request):
    return torch.device(request.param)
```

Cover the per-sensor forward path specifically (where the masking bug occurred).

**Estimated cost.** ~20 lines fixture + retrofit ~3 tests to use it.

---

## Vectorize `_apply_intervention_batch`

**Location.** `crl_vehicle/training_modes/disentangled_mode.py` near bottom.

**Current.** Per-sample Python loop; each sample's noise is generated on CPU
via `apply_intervention` and copied to device. ~30% slowdown vs vanilla VAE.

**Proposed.** Rewrite as a single vectorized op:
- Generate one (B, C, W) noise tensor on-device.
- Compute per-sample signal RMS and noise RMS in batch.
- Apply 0.2 × (signal_rms / noise_rms) scaling per sample.

Catch: each of the 7 intervention types has a different generator. Either
vectorize each type separately and then `torch.gather` by per-sample
intervention id, or stick with one type per batch (simpler — pick the
intervention id once per batch, generate B samples of that type).

**When to revisit.** If GPU utilization is the bottleneck during disentangled
mode CRL — measure `nvidia-smi` utilization mid-run; if < 70%, this is worth
fixing. If utilization is fine, defer indefinitely.
