# Full diagnostic report — multiscale_v3_lowfreq_disentangled

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 595.41 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.107196 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6711 (epoch 38) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 37 | 0.8156 | 0.6399 | 0.7578 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 17 | 0.8385 | 0.6660 | 0.7673 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 34 | 0.8072 | 0.6396 | 0.7633 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 22 | 0.8548 | 0.6687 | 0.7683 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.8072 | 0.5735 | 0.5735 | 0.7229 |
| linear_fullz__crl_best | focal | 32,674 | 0.6928 | 0.4294 | 0.4294 | 0.4490 |
| linear_fullz__crl_best | iobt | 1,575 | 0.5971 | 0.2955 | 0.5911 | 0.4136 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8525 | 0.6033 | 0.6033 | 0.7322 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.7928 | 0.4743 | 0.4743 | 0.4783 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.5888 | 0.3058 | 0.6116 | 0.4619 |
| linear_signal__crl_best | full | 91,325 | 0.8162 | 0.5752 | 0.5752 | 0.7204 |
| linear_signal__crl_best | focal | 32,674 | 0.7136 | 0.4373 | 0.4373 | 0.4496 |
| linear_signal__crl_best | iobt | 1,575 | 0.6074 | 0.2930 | 0.5859 | 0.4340 |
| linear_signal__crl_best_aux_type | full | 91,325 | 0.8310 | 0.6046 | 0.6046 | 0.7317 |
| linear_signal__crl_best_aux_type | focal | 32,674 | 0.7498 | 0.4754 | 0.4754 | 0.4778 |
| linear_signal__crl_best_aux_type | iobt | 1,575 | 0.5378 | 0.2921 | 0.5842 | 0.4414 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.397 | 0.437 | 0.825 | 0.635 |
| linear_fullz__crl_best | focal | 0.398 | 0.444 | 0.530 | 0.347 |
| linear_fullz__crl_best | iobt | 0.000 | 0.392 | 0.000 | 0.790 |
| linear_fullz__crl_best_aux_type | full | 0.443 | 0.474 | 0.840 | 0.656 |
| linear_fullz__crl_best_aux_type | focal | 0.444 | 0.478 | 0.537 | 0.438 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.444 | 0.000 | 0.780 |
| linear_signal__crl_best | full | 0.395 | 0.440 | 0.827 | 0.640 |
| linear_signal__crl_best | focal | 0.395 | 0.443 | 0.528 | 0.382 |
| linear_signal__crl_best | iobt | 0.000 | 0.414 | 0.000 | 0.757 |
| linear_signal__crl_best_aux_type | full | 0.454 | 0.472 | 0.838 | 0.655 |
| linear_signal__crl_best_aux_type | focal | 0.455 | 0.480 | 0.525 | 0.442 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.410 | 0.000 | 0.758 |
