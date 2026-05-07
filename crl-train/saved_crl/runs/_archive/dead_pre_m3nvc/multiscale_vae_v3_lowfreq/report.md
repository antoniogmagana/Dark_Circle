# Full diagnostic report — multiscale_v3_lowfreq

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 524.68 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.202818 (epoch 97) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6434 (epoch 15) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 19 | 0.8731 | 0.6505 | 0.7598 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 10 | 0.8691 | 0.6617 | 0.7480 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 1 | 0.8553 | 0.5645 | 0.7324 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 36 | 0.8704 | 0.6687 | 0.7618 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 33 | 0.8738 | 0.6517 | 0.7610 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 24 | 0.8701 | 0.6612 | 0.7466 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.8709 | 0.6005 | 0.6005 | 0.7303 |
| linear_fullz__crl_best | focal | 29,980 | 0.8460 | 0.4386 | 0.4386 | 0.4753 |
| linear_fullz__crl_best | iobt | 1,575 | 0.7596 | 0.2933 | 0.5867 | 0.4164 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.8670 | 0.6038 | 0.6038 | 0.7067 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.8539 | 0.4267 | 0.4267 | 0.4514 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.7301 | 0.3670 | 0.7339 | 0.5781 |
| linear_ztype__crl_best | full | 88,631 | 0.8703 | 0.5986 | 0.5986 | 0.7299 |
| linear_ztype__crl_best | focal | 29,980 | 0.8449 | 0.4364 | 0.4364 | 0.4746 |
| linear_ztype__crl_best | iobt | 1,575 | 0.7577 | 0.2960 | 0.5920 | 0.4210 |
| linear_ztype__crl_best_aux_type | full | 88,631 | 0.8656 | 0.6040 | 0.6040 | 0.7076 |
| linear_ztype__crl_best_aux_type | focal | 29,980 | 0.8524 | 0.4259 | 0.4259 | 0.4513 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.7241 | 0.3696 | 0.7391 | 0.5846 |
| mlp_ztype__crl_best | full | 88,631 | 0.8480 | 0.5211 | 0.5211 | 0.7050 |
| mlp_ztype__crl_best | focal | 29,980 | 0.8080 | 0.3679 | 0.3679 | 0.4882 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.6942 | 0.2159 | 0.4318 | 0.2500 |
| mlp_ztype__crl_best_aux_type | full | 88,631 | 0.8673 | 0.6133 | 0.6133 | 0.7198 |
| mlp_ztype__crl_best_aux_type | focal | 29,980 | 0.8543 | 0.4299 | 0.4299 | 0.4653 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.7310 | 0.3775 | 0.7550 | 0.5855 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.448 | 0.476 | 0.829 | 0.649 |
| linear_fullz__crl_best | focal | 0.451 | 0.489 | 0.568 | 0.246 |
| linear_fullz__crl_best | iobt | 0.000 | 0.377 | 0.000 | 0.796 |
| linear_fullz__crl_best_aux_type | full | 0.483 | 0.488 | 0.803 | 0.641 |
| linear_fullz__crl_best_aux_type | focal | 0.490 | 0.475 | 0.467 | 0.274 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.611 | 0.000 | 0.857 |
| linear_ztype__crl_best | full | 0.438 | 0.478 | 0.829 | 0.649 |
| linear_ztype__crl_best | focal | 0.441 | 0.491 | 0.572 | 0.242 |
| linear_ztype__crl_best | iobt | 0.000 | 0.385 | 0.000 | 0.799 |
| linear_ztype__crl_best_aux_type | full | 0.484 | 0.485 | 0.804 | 0.644 |
| linear_ztype__crl_best_aux_type | focal | 0.491 | 0.470 | 0.467 | 0.275 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.621 | 0.000 | 0.858 |
| mlp_ztype__crl_best | full | 0.335 | 0.469 | 0.808 | 0.472 |
| mlp_ztype__crl_best | focal | 0.336 | 0.483 | 0.589 | 0.064 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.361 | 0.000 | 0.503 |
| mlp_ztype__crl_best_aux_type | full | 0.491 | 0.490 | 0.813 | 0.660 |
| mlp_ztype__crl_best_aux_type | focal | 0.497 | 0.476 | 0.487 | 0.259 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.627 | 0.000 | 0.883 |
