# Full diagnostic report — multiscale_filesplit_v2

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 773.54 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.163072 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.5224 (epoch 13) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 32 | 0.8775 | 0.5219 | 0.7133 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 27 | 0.8669 | 0.5129 | 0.6994 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 6 | 0.8764 | 0.5233 | 0.7174 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 18 | 0.8660 | 0.5217 | 0.7122 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 12 | 0.8792 | 0.5219 | 0.7195 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 34 | 0.8660 | 0.5132 | 0.6986 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 90,627 | 0.8637 | 0.3932 | 0.3932 | 0.6480 |
| linear_fullz__crl_best | focal | 29,078 | 0.8578 | 0.2273 | 0.2273 | 0.2720 |
| linear_fullz__crl_best | iobt | 1,874 | 0.6397 | 0.3377 | 0.6755 | 0.5595 |
| linear_fullz__crl_best_aux_type | full | 90,627 | 0.8513 | 0.3975 | 0.3975 | 0.6231 |
| linear_fullz__crl_best_aux_type | focal | 29,078 | 0.8381 | 0.2394 | 0.2394 | 0.2654 |
| linear_fullz__crl_best_aux_type | iobt | 1,874 | 0.5248 | 0.4137 | 0.8274 | 0.7690 |
| linear_ztype__crl_best | full | 90,627 | 0.8628 | 0.3925 | 0.3925 | 0.6426 |
| linear_ztype__crl_best | focal | 29,078 | 0.8571 | 0.2309 | 0.2309 | 0.2734 |
| linear_ztype__crl_best | iobt | 1,874 | 0.6317 | 0.3380 | 0.6760 | 0.5723 |
| linear_ztype__crl_best_aux_type | full | 90,627 | 0.8519 | 0.3982 | 0.3982 | 0.6233 |
| linear_ztype__crl_best_aux_type | focal | 29,078 | 0.8393 | 0.2385 | 0.2385 | 0.2635 |
| linear_ztype__crl_best_aux_type | iobt | 1,874 | 0.5316 | 0.4132 | 0.8263 | 0.7690 |
| mlp_ztype__crl_best | full | 90,627 | 0.8621 | 0.3934 | 0.3934 | 0.6449 |
| mlp_ztype__crl_best | focal | 29,078 | 0.8559 | 0.2276 | 0.2276 | 0.2679 |
| mlp_ztype__crl_best | iobt | 1,874 | 0.6320 | 0.3357 | 0.6714 | 0.5577 |
| mlp_ztype__crl_best_aux_type | full | 90,627 | 0.8511 | 0.3913 | 0.3913 | 0.6336 |
| mlp_ztype__crl_best_aux_type | focal | 29,078 | 0.8379 | 0.2324 | 0.2324 | 0.2686 |
| mlp_ztype__crl_best_aux_type | iobt | 1,874 | 0.5248 | 0.4062 | 0.8124 | 0.7305 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.026 | 0.198 | 0.777 | 0.572 |
| linear_fullz__crl_best | focal | 0.027 | 0.165 | 0.405 | 0.313 |
| linear_fullz__crl_best | iobt | 0.000 | 0.452 | 0.000 | 0.899 |
| linear_fullz__crl_best_aux_type | full | 0.021 | 0.249 | 0.772 | 0.549 |
| linear_fullz__crl_best_aux_type | focal | 0.021 | 0.199 | 0.395 | 0.342 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.761 | 0.000 | 0.893 |
| linear_ztype__crl_best | full | 0.027 | 0.201 | 0.775 | 0.568 |
| linear_ztype__crl_best | focal | 0.027 | 0.168 | 0.411 | 0.318 |
| linear_ztype__crl_best | iobt | 0.000 | 0.456 | 0.000 | 0.896 |
| linear_ztype__crl_best_aux_type | full | 0.021 | 0.252 | 0.773 | 0.548 |
| linear_ztype__crl_best_aux_type | focal | 0.021 | 0.204 | 0.390 | 0.339 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.760 | 0.000 | 0.892 |
| mlp_ztype__crl_best | full | 0.030 | 0.196 | 0.776 | 0.572 |
| mlp_ztype__crl_best | focal | 0.030 | 0.163 | 0.397 | 0.320 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.450 | 0.000 | 0.893 |
| mlp_ztype__crl_best_aux_type | full | 0.020 | 0.213 | 0.777 | 0.554 |
| mlp_ztype__crl_best_aux_type | focal | 0.020 | 0.156 | 0.405 | 0.348 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.711 | 0.000 | 0.914 |
