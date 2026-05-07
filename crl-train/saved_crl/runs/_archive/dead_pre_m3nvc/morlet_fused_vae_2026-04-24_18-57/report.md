# Full diagnostic report — 2026-04-24_18-57-44

## CRL pre-training

- frontend: `morlet_fused`, d_z=24, d_model=64, n_layers=2

- elapsed: 77.38 min

- total epochs recorded: 82

- **best val_ref_elbo:** 2.311411 (epoch 56) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4522 (epoch 34) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 40 | 0.6985 | 0.2910 | 0.3937 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 7 | 0.6972 | 0.3944 | 0.4597 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 49 | 0.6974 | 0.3229 | 0.4400 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 42 | 0.6859 | 0.3973 | 0.4579 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 36 | 0.7040 | 0.3335 | 0.3936 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 23 | 0.6923 | 0.4097 | 0.4518 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.7517 | 0.4217 | 0.4217 | 0.4787 |
| linear_fullz__crl_best | focal | 14,534 | 0.7899 | 0.2734 | 0.2734 | 0.3374 |
| linear_fullz__crl_best | iobt | 936 | 0.7253 | 0.3198 | 0.6397 | 0.5153 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.7329 | 0.4642 | 0.4642 | 0.5057 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.7856 | 0.3060 | 0.3060 | 0.3284 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.7356 | 0.4130 | 0.8260 | 0.7536 |
| linear_ztype__crl_best | full | 21,436 | 0.7456 | 0.3774 | 0.3774 | 0.4575 |
| linear_ztype__crl_best | focal | 14,534 | 0.7841 | 0.2402 | 0.2402 | 0.3015 |
| linear_ztype__crl_best | iobt | 936 | 0.7189 | 0.3521 | 0.7042 | 0.6490 |
| linear_ztype__crl_best_aux_type | full | 21,436 | 0.7373 | 0.4272 | 0.4272 | 0.4894 |
| linear_ztype__crl_best_aux_type | focal | 14,534 | 0.7901 | 0.2397 | 0.2397 | 0.2954 |
| linear_ztype__crl_best_aux_type | iobt | 936 | 0.7366 | 0.4535 | 0.9069 | 0.8696 |
| mlp_ztype__crl_best | full | 21,436 | 0.7447 | 0.3339 | 0.3339 | 0.4239 |
| mlp_ztype__crl_best | focal | 14,534 | 0.7837 | 0.1215 | 0.1215 | 0.2209 |
| mlp_ztype__crl_best | iobt | 936 | 0.7189 | 0.1848 | 0.3695 | 0.5266 |
| mlp_ztype__crl_best_aux_type | full | 21,436 | 0.7275 | 0.4072 | 0.4072 | 0.4658 |
| mlp_ztype__crl_best_aux_type | focal | 14,534 | 0.7814 | 0.2039 | 0.2039 | 0.2658 |
| mlp_ztype__crl_best_aux_type | iobt | 936 | 0.7289 | 0.3868 | 0.7735 | 0.7472 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.432 | 0.374 | 0.663 | 0.217 |
| linear_fullz__crl_best | focal | 0.446 | 0.372 | 0.222 | 0.053 |
| linear_fullz__crl_best | iobt | 0.000 | 0.397 | 0.000 | 0.882 |
| linear_fullz__crl_best_aux_type | full | 0.261 | 0.451 | 0.714 | 0.431 |
| linear_fullz__crl_best_aux_type | focal | 0.265 | 0.431 | 0.345 | 0.183 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.721 | 0.000 | 0.931 |
| linear_ztype__crl_best | full | 0.229 | 0.436 | 0.666 | 0.179 |
| linear_ztype__crl_best | focal | 0.233 | 0.417 | 0.249 | 0.061 |
| linear_ztype__crl_best | iobt | 0.000 | 0.685 | 0.000 | 0.723 |
| linear_ztype__crl_best_aux_type | full | 0.112 | 0.455 | 0.715 | 0.426 |
| linear_ztype__crl_best_aux_type | focal | 0.113 | 0.425 | 0.363 | 0.057 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.875 | 0.000 | 0.938 |
| mlp_ztype__crl_best | full | 0.017 | 0.387 | 0.654 | 0.279 |
| mlp_ztype__crl_best | focal | 0.017 | 0.362 | 0.106 | 0.001 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.684 | 0.000 | 0.055 |
| mlp_ztype__crl_best_aux_type | full | 0.098 | 0.445 | 0.685 | 0.402 |
| mlp_ztype__crl_best_aux_type | focal | 0.098 | 0.419 | 0.244 | 0.054 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.785 | 0.000 | 0.762 |
