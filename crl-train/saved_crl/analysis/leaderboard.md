# CRL Run Leaderboard

| Run | Frontend | Mode | Ep | pres_MCC | pres_BalAcc | pres_F1 | pres_probe | min_pres_F1 | type_F1 | type_probe | min_type_F1 | ELBO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-03_05-03-14 | multiscale | disentangled | 100 | 0.2717 | 0.6126 | 0.8438 | linear_signal | 0.786 | 0.6702 | linear_signal_aux | 0.4365 | 0.2524 |
| ✓ 2026-05-03_15-26-22 |  |  | 0 | 0.4388 | 0.6837 | 0.8714 | mlp_ztype | 0.8224 | 0.6551 | linear_fullz_aux | 0.4572 |  |
| 2026-05-03_05-02-44 | multiscale | vae | 51 | 0.4754 | 0.7461 | 0.867 | mlp_ztype | 0.6166 | 0.6391 | linear_ztype | 0.4554 | 0.5526 |

- ✓ = shippable (pres_f1 ≥ 0.85, type_f1 ≥ 0.65)
- ⚠ = diverged (val_ref_elbo > 50.0)