# Multiscale CRL: VAE vs Disentangled — basics

Both runs use the **multiscale frontend** and the same encoder/decoder backbone. The only science variable is the **training objective** (VAE vs Disentangled).

## Shared backbone (same for both)

- **Frontend**: multiscale — three parallel learned Conv1D branches at kernel sizes `[9, 19, 39]`, fused via 1×1 projection to `d_model = 64`, 32 tokens.
- **Latent dim** `d_z = 24`, **model dim** `d_model = 64`
- **Encoder/decoder**: 2-layer transformer, 4 heads, fused token length 32
- **Optimization**: Adam, `lr = 3e-4 → 1e-4` (cosine), `wd = 1e-4`, batch size 64, up to 100 epochs, early-stop patience 25 monitoring `val_ref_elbo`
- **Downstream evaluation**: `linear_fullz` probe (frontend-agnostic — reads the full latent, not the signal/env split) over 50 epochs; reports `val_pres_f1` and `val_type_f1`
- **Checkpoint metric**: β-invariant reference ELBO, `val_ref_elbo = val_recon + val_raw_kl`, evaluated at β = 1

## What differs

| Aspect | Multiscale + VAE | Multiscale + Disentangled |
|---|---|---|
| `training_mode` | `vae` | `disentangled` |
| Latent partition | flat `z` with named blocks (pres / type / prox / env / free) | explicit signal / env split (`SplitLatentSpace`) |
| Loss components | `L_recon + β·L_KL + λ_p·L_pres + λ_t·L_type` | `L_recon + β·L_KL + L_align + L_stable + L_invar` |
| Cross-modal alignment | implicit (shared encoder) | explicit `‖μ_signal,audio − μ_signal,seismic‖²` |
| Env stability | none | `‖μ_env,t − μ_env,t+1‖²` on consecutive windows |
| Signal invariance | implicit | `‖μ_signal,clean − μ_signal,intervened‖²` under noise interventions |
| Aux heads | presence + type read μ | presence + type read full d_signal block |
| Prior | `StandardPrior` (configurable to `ConditionalPrior` for iVAE) | `StandardPrior` only (disentangled fixes the prior) |

## Headline numbers (this run)

| Metric | VAE | Disentangled |
|---|---:|---:|
| Best CRL `val_ref_elbo` (lower is better) | 0.553 | **0.252** |
| Best CRL `val_aux_type_f1` (epoch) | 0.607 (ep 4) | **0.695 (ep 32)** |
| Downstream best `val_pres_f1` (`linear_fullz`) | TBD on rerun | TBD on rerun |
| Downstream best `val_type_f1` (`linear_fullz`) | TBD on rerun | TBD on rerun |
| CRL epochs run | 51 | 100 |

Disentangled wins on the β-invariant generative metric and on the auxiliary type-F1 proxy during pretraining. Downstream probe numbers are regenerated alongside the figures by `_make_figures.py`; see `_summary.json` for the latest values.
