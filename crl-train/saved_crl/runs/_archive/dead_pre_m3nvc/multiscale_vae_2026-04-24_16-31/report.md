# Full diagnostic report — 2026-04-24_16-31-36

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 84.56 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.173641 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4638 (epoch 5) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 45 | 0.8475 | 0.4606 | 0.6011 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 3 | 0.8562 | 0.4729 | 0.5810 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 5 | 0.8491 | 0.2945 | 0.5104 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 29 | 0.8592 | 0.4489 | 0.5551 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 48 | 0.8862 | 0.4663 | 0.6054 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 2 | 0.8509 | 0.4599 | 0.5686 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.8550 | 0.3326 | 0.3326 | 0.4768 |
| linear_fullz__crl_best | focal | 14,534 | 0.8442 | 0.2182 | 0.2182 | 0.2755 |
| linear_fullz__crl_best | iobt | 936 | 0.7828 | 0.3067 | 0.6135 | 0.5733 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.8078 | 0.3880 | 0.3880 | 0.4591 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.8095 | 0.2442 | 0.2442 | 0.2439 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.7482 | 0.3428 | 0.6857 | 0.6860 |
| linear_ztype__crl_best | full | 21,436 | 0.8113 | 0.3288 | 0.3288 | 0.4810 |
| linear_ztype__crl_best | focal | 14,534 | 0.8008 | 0.2181 | 0.2181 | 0.2829 |
| linear_ztype__crl_best | iobt | 936 | 0.7027 | 0.2922 | 0.5845 | 0.5491 |
| linear_ztype__crl_best_aux_type | full | 21,436 | 0.8131 | 0.3644 | 0.3644 | 0.4391 |
| linear_ztype__crl_best_aux_type | focal | 14,534 | 0.8129 | 0.2140 | 0.2140 | 0.2140 |
| linear_ztype__crl_best_aux_type | iobt | 936 | 0.7513 | 0.3511 | 0.7022 | 0.6973 |
| mlp_ztype__crl_best | full | 21,436 | 0.8129 | 0.2950 | 0.2950 | 0.4886 |
| mlp_ztype__crl_best | focal | 14,534 | 0.8025 | 0.1984 | 0.1984 | 0.3039 |
| mlp_ztype__crl_best | iobt | 936 | 0.7045 | 0.1913 | 0.3826 | 0.3671 |
| mlp_ztype__crl_best_aux_type | full | 21,436 | 0.8167 | 0.3808 | 0.3808 | 0.4439 |
| mlp_ztype__crl_best_aux_type | focal | 14,534 | 0.8157 | 0.2134 | 0.2134 | 0.2208 |
| mlp_ztype__crl_best_aux_type | iobt | 936 | 0.7543 | 0.3711 | 0.7423 | 0.7375 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.019 | 0.146 | 0.641 | 0.524 |
| linear_fullz__crl_best | focal | 0.019 | 0.099 | 0.418 | 0.337 |
| linear_fullz__crl_best | iobt | 0.000 | 0.469 | 0.000 | 0.758 |
| linear_fullz__crl_best_aux_type | full | 0.038 | 0.358 | 0.651 | 0.505 |
| linear_fullz__crl_best_aux_type | focal | 0.038 | 0.335 | 0.274 | 0.331 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.609 | 0.000 | 0.762 |
| linear_ztype__crl_best | full | 0.017 | 0.130 | 0.645 | 0.524 |
| linear_ztype__crl_best | focal | 0.017 | 0.087 | 0.431 | 0.338 |
| linear_ztype__crl_best | iobt | 0.000 | 0.414 | 0.000 | 0.755 |
| linear_ztype__crl_best_aux_type | full | 0.038 | 0.286 | 0.616 | 0.519 |
| linear_ztype__crl_best_aux_type | focal | 0.038 | 0.250 | 0.207 | 0.361 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.636 | 0.000 | 0.769 |
| mlp_ztype__crl_best | full | 0.016 | 0.000 | 0.640 | 0.523 |
| mlp_ztype__crl_best | focal | 0.016 | 0.000 | 0.459 | 0.318 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.000 | 0.000 | 0.765 |
| mlp_ztype__crl_best_aux_type | full | 0.034 | 0.344 | 0.623 | 0.523 |
| mlp_ztype__crl_best_aux_type | focal | 0.034 | 0.318 | 0.167 | 0.335 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.700 | 0.000 | 0.785 |
