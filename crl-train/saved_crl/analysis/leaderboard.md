# CRL Run Leaderboard

| Run | Frontend | Phase | Stage2 | Ep | pres_f1 | type_f1 | ELBO | min-ds F1 | worst | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| v3_lowfreq | multiscale | False | False | 100 | 0.875 | 0.6565 | 0.2028 | 0.4364 | focal | 0.5088 |
| v3_lowfreq | multiscale | False | False | 100 | 0.8612 | 0.6499 | 0.1072 | 0.4294 | focal | 0.4212 |
| 2026-04-28_23-12-59 | multiscale | False | False | 100 | 0.8777 | 0.6351 | 0.214 | 0.3359 | focal | 0.0 |
| v2 | multiscale | False | False | 100 | 0.8767 | 0.6082 | 0.1799 | 0.4071 | focal | 0.5075 |
| filesplit_v2 | multiscale | False | False | 100 | 0.8924 | 0.5248 | 0.1631 | 0.2309 | focal | 0.5376 |
| phase_v1_diag | morlet_per_sensor | True | False | 78 | 0.7884 | 0.4977 | 2.2769 | 0.3568 | focal | 0.3121 |
| 2026-04-25_09-41 | multiscale | False | False | 68 | 0.8859 | 0.4778 | 0.1438 | 0.2409 | focal | 0.3755 |
| 2026-04-24_16-31 | multiscale | False | False | 100 | 0.8837 | 0.4718 | 0.1736 | 0.2181 | focal | 0.4394 |
| 2026-04-24_16-17 | multiscale | False | False | 35 | 0.7519 | 0.4029 |  | 0.1708 | focal | 0.0566 |
| 2026-04-25_10-06 | morlet_per_sensor | True | False | 96 | 0.8707 | 0.3701 | 1.9765 | 0.2535 | focal | 0.2699 |
| 2026-04-24_18-57 | morlet_fused | True | False | 82 | 0.8192 | 0.3489 | 2.3114 | 0.2402 | focal | 0.4423 |
| framework_smoke | multiscale |  | False | 2 | 0.5774 | 0.3164 | 8.6386 | 0.0203 | focal | 0.1226 |

- ✓ = shippable (pres_f1 ≥ 0.85, type_f1 ≥ 0.7)
- ⚠ = diverged (val_ref_elbo > 50.0)