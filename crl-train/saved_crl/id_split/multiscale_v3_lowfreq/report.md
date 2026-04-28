# Full diagnostic report — multiscale_v3_lowfreq

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 611.27 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.162046 (epoch 99) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6509 (epoch 28) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 28 | 0.8762 | 0.6223 | 0.7402 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 12 | 0.8593 | 0.6566 | 0.7520 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 25 | 0.8728 | 0.6041 | 0.7315 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 26 | 0.8615 | 0.6539 | 0.7570 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 22 | 0.8619 | 0.6216 | 0.7389 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 43 | 0.8653 | 0.6570 | 0.7529 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.8598 | 0.5776 | 0.5776 | 0.7218 |
| linear_fullz__crl_best | focal | 32,674 | 0.8364 | 0.4601 | 0.4601 | 0.4767 |
| linear_fullz__crl_best | iobt | 1,575 | 0.6898 | 0.2752 | 0.5504 | 0.3978 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8655 | 0.6098 | 0.6098 | 0.7259 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8596 | 0.4630 | 0.4630 | 0.4635 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.6846 | 0.3271 | 0.6542 | 0.5019 |
| linear_ztype__crl_best | full | 91,325 | 0.8762 | 0.5783 | 0.5783 | 0.7234 |
| linear_ztype__crl_best | focal | 32,674 | 0.8600 | 0.4608 | 0.4608 | 0.4774 |
| linear_ztype__crl_best | iobt | 1,575 | 0.7426 | 0.2742 | 0.5484 | 0.3950 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.8589 | 0.6091 | 0.6091 | 0.7253 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.8531 | 0.4636 | 0.4636 | 0.4638 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.6525 | 0.3245 | 0.6491 | 0.5019 |
| mlp_ztype__crl_best | full | 91,325 | 0.8720 | 0.5619 | 0.5619 | 0.7182 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8538 | 0.4496 | 0.4496 | 0.4716 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.7282 | 0.2653 | 0.5306 | 0.3885 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.8617 | 0.6119 | 0.6119 | 0.7340 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.8559 | 0.4753 | 0.4753 | 0.4782 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.6686 | 0.3258 | 0.6516 | 0.4842 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.396 | 0.443 | 0.830 | 0.641 |
| linear_fullz__crl_best | focal | 0.397 | 0.456 | 0.589 | 0.398 |
| linear_fullz__crl_best | iobt | 0.000 | 0.345 | 0.000 | 0.756 |
| linear_fullz__crl_best_aux_type | full | 0.474 | 0.481 | 0.835 | 0.649 |
| linear_fullz__crl_best_aux_type | focal | 0.482 | 0.480 | 0.493 | 0.397 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.504 | 0.000 | 0.805 |
| linear_ztype__crl_best | full | 0.401 | 0.438 | 0.831 | 0.643 |
| linear_ztype__crl_best | focal | 0.402 | 0.452 | 0.588 | 0.402 |
| linear_ztype__crl_best | iobt | 0.000 | 0.340 | 0.000 | 0.757 |
| linear_ztype__crl_best_aux_type | full | 0.472 | 0.482 | 0.834 | 0.649 |
| linear_ztype__crl_best_aux_type | focal | 0.480 | 0.481 | 0.495 | 0.399 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.505 | 0.000 | 0.793 |
| mlp_ztype__crl_best | full | 0.350 | 0.431 | 0.829 | 0.638 |
| mlp_ztype__crl_best | focal | 0.351 | 0.445 | 0.596 | 0.407 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.326 | 0.000 | 0.735 |
| mlp_ztype__crl_best_aux_type | full | 0.473 | 0.479 | 0.842 | 0.653 |
| mlp_ztype__crl_best_aux_type | focal | 0.480 | 0.481 | 0.529 | 0.411 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.477 | 0.000 | 0.826 |
