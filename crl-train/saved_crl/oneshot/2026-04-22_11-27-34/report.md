# Full diagnostic report — 2026-04-22_11-27-34

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 168.55 min

- total epochs recorded: 100

- **best val_ref_elbo:** 2.078633 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.5035 (epoch 13) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 47 | 0.7888 | 0.4663 | 0.6174 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 17 | 0.7440 | 0.4921 | 0.5956 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 47 | 0.7990 | 0.3841 | 0.6060 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 12 | 0.7485 | 0.4784 | 0.6026 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 36 | 0.7922 | 0.4610 | 0.5945 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 9 | 0.7467 | 0.4750 | 0.5917 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7182 | 0.2462 | 0.2462 | 0.2810 |
| linear_fullz__crl_best | focal | 9,228 | 0.7178 | 0.2084 | 0.4168 | 0.3095 |
| linear_fullz__crl_best | iobt | 3,204 | 0.7192 | 0.1308 | 0.2616 | 0.1897 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.5313 | 0.2753 | 0.2753 | 0.3072 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.5182 | 0.2211 | 0.4423 | 0.3312 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.5686 | 0.1717 | 0.3433 | 0.2304 |
| linear_ztype__crl_best | full | 12,432 | 0.7124 | 0.2465 | 0.2465 | 0.3094 |
| linear_ztype__crl_best | focal | 9,228 | 0.7115 | 0.2233 | 0.4467 | 0.3704 |
| linear_ztype__crl_best | iobt | 3,204 | 0.7151 | 0.0887 | 0.1774 | 0.1146 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.5272 | 0.2412 | 0.2412 | 0.2462 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.5129 | 0.1658 | 0.3316 | 0.2342 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.5675 | 0.1986 | 0.3973 | 0.2846 |
| mlp_ztype__crl_best | full | 12,432 | 0.7272 | 0.2319 | 0.2319 | 0.3798 |
| mlp_ztype__crl_best | focal | 9,228 | 0.7277 | 0.2492 | 0.4984 | 0.4932 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.7258 | 0.0173 | 0.0345 | 0.0170 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.5357 | 0.2180 | 0.2180 | 0.2409 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.5233 | 0.1764 | 0.3528 | 0.2631 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.5709 | 0.1236 | 0.2473 | 0.1700 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.381 | 0.174 | 0.357 | 0.073 |
| linear_fullz__crl_best | focal | 0.423 | 0.000 | 0.411 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.315 | 0.000 | 0.208 |
| linear_fullz__crl_best_aux_type | full | 0.371 | 0.138 | 0.409 | 0.182 |
| linear_fullz__crl_best_aux_type | focal | 0.408 | 0.000 | 0.477 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.305 | 0.000 | 0.382 |
| linear_ztype__crl_best | full | 0.398 | 0.117 | 0.393 | 0.077 |
| linear_ztype__crl_best | focal | 0.445 | 0.000 | 0.449 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.161 | 0.000 | 0.194 |
| linear_ztype__crl_best_aux_type | full | 0.197 | 0.153 | 0.384 | 0.231 |
| linear_ztype__crl_best_aux_type | focal | 0.216 | 0.000 | 0.447 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.419 | 0.000 | 0.376 |
| mlp_ztype__crl_best | full | 0.374 | 0.026 | 0.489 | 0.038 |
| mlp_ztype__crl_best | focal | 0.435 | 0.000 | 0.562 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.030 | 0.000 | 0.039 |
| mlp_ztype__crl_best_aux_type | full | 0.252 | 0.121 | 0.352 | 0.147 |
| mlp_ztype__crl_best_aux_type | focal | 0.294 | 0.000 | 0.411 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.342 | 0.000 | 0.152 |
