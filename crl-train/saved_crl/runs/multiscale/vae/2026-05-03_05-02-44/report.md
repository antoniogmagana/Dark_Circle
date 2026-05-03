# Full diagnostic report — 2026-05-03_05-02-44

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 300.76 min

- total epochs recorded: 51

- **best val_ref_elbo:** 0.552606 (epoch 25) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6071 (epoch 4) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 34 | 0.8512 | 0.6539 | 0.7343 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 12 | 0.8445 | 0.6298 | 0.6997 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 15 | 0.8484 | 0.6592 | 0.7600 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 4 | 0.8458 | 0.6347 | 0.7142 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 11 | 0.8604 | 0.6561 | 0.7390 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 30 | 0.8444 | 0.6288 | 0.6970 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.8463 | 0.5988 | 0.5988 | 0.7094 |
| linear_fullz__crl_best | focal | 29,980 | 0.8265 | 0.4420 | 0.4420 | 0.4934 |
| linear_fullz__crl_best | iobt | 1,575 | 0.5792 | 0.3199 | 0.6399 | 0.4545 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.8283 | 0.5816 | 0.5816 | 0.6611 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.7970 | 0.3924 | 0.3924 | 0.4397 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.4850 | 0.3277 | 0.6553 | 0.5167 |
| linear_ztype__crl_best | full | 88,631 | 0.8344 | 0.5975 | 0.5975 | 0.7065 |
| linear_ztype__crl_best | focal | 29,980 | 0.8103 | 0.4461 | 0.4461 | 0.4934 |
| linear_ztype__crl_best | iobt | 1,575 | 0.5399 | 0.3122 | 0.6243 | 0.4554 |
| linear_ztype__crl_best_aux_type | full | 88,631 | 0.8285 | 0.5811 | 0.5811 | 0.6613 |
| linear_ztype__crl_best_aux_type | focal | 29,980 | 0.7974 | 0.3872 | 0.3872 | 0.4336 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.4860 | 0.3149 | 0.6299 | 0.5102 |
| mlp_ztype__crl_best | full | 88,631 | 0.8307 | 0.5948 | 0.5948 | 0.7224 |
| mlp_ztype__crl_best | focal | 29,980 | 0.8055 | 0.4240 | 0.4240 | 0.4963 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.5303 | 0.3275 | 0.6550 | 0.4191 |
| mlp_ztype__crl_best_aux_type | full | 88,631 | 0.8303 | 0.5885 | 0.5885 | 0.6772 |
| mlp_ztype__crl_best_aux_type | focal | 29,980 | 0.8000 | 0.3881 | 0.3881 | 0.4453 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.4936 | 0.3529 | 0.7059 | 0.5223 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.471 | 0.491 | 0.800 | 0.634 |
| linear_fullz__crl_best | focal | 0.474 | 0.494 | 0.557 | 0.242 |
| linear_fullz__crl_best | iobt | 0.000 | 0.461 | 0.000 | 0.819 |
| linear_fullz__crl_best_aux_type | full | 0.497 | 0.502 | 0.763 | 0.564 |
| linear_fullz__crl_best_aux_type | focal | 0.505 | 0.502 | 0.391 | 0.172 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.530 | 0.000 | 0.780 |
| linear_ztype__crl_best | full | 0.470 | 0.492 | 0.798 | 0.630 |
| linear_ztype__crl_best | focal | 0.474 | 0.495 | 0.560 | 0.255 |
| linear_ztype__crl_best | iobt | 0.000 | 0.458 | 0.000 | 0.790 |
| linear_ztype__crl_best_aux_type | full | 0.499 | 0.502 | 0.765 | 0.559 |
| linear_ztype__crl_best_aux_type | focal | 0.506 | 0.503 | 0.373 | 0.167 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.521 | 0.000 | 0.739 |
| mlp_ztype__crl_best | full | 0.465 | 0.469 | 0.808 | 0.637 |
| mlp_ztype__crl_best | focal | 0.468 | 0.476 | 0.563 | 0.188 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.415 | 0.000 | 0.895 |
| mlp_ztype__crl_best_aux_type | full | 0.499 | 0.499 | 0.781 | 0.575 |
| mlp_ztype__crl_best_aux_type | focal | 0.508 | 0.498 | 0.392 | 0.155 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.545 | 0.000 | 0.866 |
