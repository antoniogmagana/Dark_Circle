# Multiscale vs. Morlet base CRL models — basics

Both runs are causal-representation-learning (CRL) pretraining, identical *except* for the audio/seismic frontend that turns raw waveforms into per-token features.

## Shared backbone (same for both)

- **Latent dim** `d_z = 24`, **model dim** `d_model = 64`
- **Encoder/decoder**: 2-layer transformer, 4 heads, fused token length 32
- **Training mode**: VAE with standard Gaussian prior, β-annealed (`beta_step = 0.02`, KL floor 0.01, KL target 0.5)
- **Auxiliary heads** during pretraining: presence classifier, vehicle-type classifier, intervention loss (`λ_interv = 1.0`, `λ_aux_pres = 1.0`, `λ_aux_type = 1.0`)
- **Optimization**: Adam, `lr = 3e-4 → 1e-4` (cosine), `wd = 1e-4`, batch size 64, up to 100 epochs, early-stop patience 25 monitoring `val_ref_elbo`
- **Downstream evaluation**: linear probe on `z_type` over 50 epochs; reports `val_pres_f1` and `val_type_f1`
- **Dual CRL checkpoint**: `crl_best.pth` selected by `val_ref_elbo` (β-invariant), `crl_best_aux_type.pth` selected by `val_aux_type_f1`

## What differs

| Aspect | Multiscale frontend | Morlet (per-sensor) frontend |
|---|---|---|
| `frontend_type` | `multiscale` | `morlet_per_sensor` |
| Mechanism | Stack of **learned** strided 1-D convs at multiple receptive fields, fused into 32 tokens | Bank of **fixed-frequency** Morlet wavelets per sensor (audio: 20 Hz–8 kHz, seismic: 2–40 Hz), 32 tokens, `w0 = 6`, 3 receptive cycles |
| Phase channel | n/a | **Enabled** (`morlet_use_phase = true`) — magnitude + phase per wavelet |
| Pool stride | 16 | 64 |
| Inductive bias | Data-driven; learns its own filters | Time-frequency prior; physics-shaped filters per sensor band |

## Headline numbers (this run)

| Metric | Multiscale | Morlet (per-sensor) |
|---|---:|---:|
| Best CRL `val_ref_elbo` (lower is better) | **0.170** | 1.710 |
| Best CRL `val_aux_type_f1` (epoch) | **0.480 (ep 27)** | 0.319 (ep 25) |
| Downstream best `val_pres_f1` | **0.878** | 0.855 |
| Downstream best `val_type_f1` | **0.475** | 0.375 |
| CRL epochs run / patience target | 100 / 25 on `val_ref_elbo` | 83 / 25 on `val_ref_elbo` |

Multiscale wins on every comparable metric in this configuration. The gap is dominated by **vehicle-type F1** (0.475 vs. 0.375) — presence detection is roughly tied, but type discrimination is where the learned multiscale features outperform the fixed Morlet bank.
