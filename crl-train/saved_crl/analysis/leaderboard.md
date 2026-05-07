# CRL Run Leaderboard

| Run | Frontend | Phase | Stage2 | Ep | pres_f1 | type_f1 | ELBO | min-ds F1 | worst | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| ✓ 2026-04-29_13-26-17 | multiscale | False | False | 100 | 0.8742 | 0.6589 | 0.1926 |  |  |  |
| ✓ v3_lowfreq | multiscale | False | False | 100 | 0.875 | 0.6565 | 0.2028 |  |  |  |
| v3_lowfreq | multiscale | False | False | 100 | 0.8612 | 0.6499 | 0.1072 |  |  |  |
| 2026-05-03_05-03-14 | multiscale | False | False | 100 | 0.8433 | 0.6462 | 0.2524 | 0.4354 | focal | 0.2959 |
| 2026-05-03_05-02-44 | multiscale | False | False | 51 | 0.8625 | 0.6391 | 0.5526 | 0.4554 | focal | 0.4735 |
| 2026-04-28_23-12-59 | multiscale | False | False | 100 | 0.8777 | 0.6351 | 0.214 |  |  |  |
| v2 | multiscale | False | False | 100 | 0.8767 | 0.6082 | 0.1799 |  |  |  |
| filesplit_v2 | multiscale | False | False | 100 | 0.8924 | 0.5248 | 0.1631 |  |  |  |
| phase_v1_diag | morlet_per_sensor | True | False | 78 | 0.7884 | 0.4977 | 2.2769 |  |  |  |
| 2026-05-04_05-49-26 | morlet_per_sensor | False | False | 78 | 0.7432 | 0.4871 | 4.46 | 0.3357 | focal | 0.2288 |
| 2026-04-25_09-41 | multiscale | False | False | 68 | 0.8859 | 0.4778 | 0.1438 |  |  |  |
| 2026-04-24_16-31 | multiscale | False | False | 100 | 0.8837 | 0.4718 | 0.1736 |  |  |  |
| 2026-04-24_16-17 | multiscale | False | False | 35 | 0.7519 | 0.4029 |  |  |  |  |
| 2026-04-25_10-06 | morlet_per_sensor | True | False | 96 | 0.8707 | 0.3701 | 1.9765 |  |  |  |
| 2026-04-24_18-57 | morlet_fused | True | False | 82 | 0.8192 | 0.3489 | 2.3114 |  |  |  |

- ✓ = shippable (pres_f1 ≥ 0.85, type_f1 ≥ 0.65)
- ⚠ = diverged (val_ref_elbo > 50.0)