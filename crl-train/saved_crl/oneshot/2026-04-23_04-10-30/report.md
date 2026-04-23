# Full diagnostic report — 2026-04-23_04-10-30

## CRL pre-training

- frontend: `morlet`, d_z=24, d_model=64, n_layers=2

- elapsed: 138.19 min

- total epochs recorded: 73

- **best val_ref_elbo:** 778.086488 (epoch 47) → `crl_best.pth`

- **best val_aux_type_f1:** 0.3412 (epoch 33) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 13 | 0.7880 | 0.4089 | 0.5635 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 7 | 0.7887 | 0.3861 | 0.5542 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 2 | 0.4675 | 0.3611 | 0.5463 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 43 | 0.7891 | 0.4101 | 0.5543 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 3 | 0.7886 | 0.4152 | 0.5626 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 13 | 0.4593 | 0.3820 | 0.5344 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7173 | 0.1731 | 0.1731 | 0.2887 |
| linear_fullz__crl_best | focal | 9,228 | 0.7265 | 0.1706 | 0.3413 | 0.3549 |
| linear_fullz__crl_best | iobt | 3,204 | 0.6891 | 0.0687 | 0.1373 | 0.0769 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.3302 | 0.1881 | 0.1881 | 0.2893 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.3278 | 0.1959 | 0.3918 | 0.3664 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.3376 | 0.0385 | 0.0770 | 0.0429 |
| linear_ztype__crl_best | full | 12,432 | 0.7167 | 0.2284 | 0.2284 | 0.3229 |
| linear_ztype__crl_best | focal | 9,228 | 0.7261 | 0.1880 | 0.3760 | 0.3807 |
| linear_ztype__crl_best | iobt | 3,204 | 0.6882 | 0.1166 | 0.2332 | 0.1380 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.7136 | 0.2241 | 0.2241 | 0.3162 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.7202 | 0.2010 | 0.4020 | 0.3811 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.6935 | 0.0922 | 0.1844 | 0.1087 |
| mlp_ztype__crl_best | full | 12,432 | 0.3494 | 0.1990 | 0.1990 | 0.3140 |
| mlp_ztype__crl_best | focal | 9,228 | 0.3512 | 0.1931 | 0.3863 | 0.3811 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.3437 | 0.0847 | 0.1694 | 0.0993 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.7143 | 0.2763 | 0.2763 | 0.3080 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.7207 | 0.1668 | 0.3337 | 0.3299 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.6950 | 0.1905 | 0.3811 | 0.2378 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.075 | 0.036 | 0.501 | 0.080 |
| linear_fullz__crl_best | focal | 0.076 | 0.000 | 0.607 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.053 | 0.000 | 0.222 |
| linear_fullz__crl_best_aux_type | full | 0.222 | 0.034 | 0.456 | 0.041 |
| linear_fullz__crl_best_aux_type | focal | 0.225 | 0.000 | 0.558 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.048 | 0.000 | 0.106 |
| linear_ztype__crl_best | full | 0.143 | 0.102 | 0.502 | 0.167 |
| linear_ztype__crl_best | focal | 0.145 | 0.000 | 0.607 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.200 | 0.000 | 0.267 |
| linear_ztype__crl_best_aux_type | full | 0.227 | 0.097 | 0.468 | 0.104 |
| linear_ztype__crl_best_aux_type | focal | 0.231 | 0.000 | 0.573 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.182 | 0.000 | 0.186 |
| mlp_ztype__crl_best | full | 0.175 | 0.000 | 0.498 | 0.123 |
| mlp_ztype__crl_best | focal | 0.176 | 0.000 | 0.596 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.000 | 0.000 | 0.339 |
| mlp_ztype__crl_best_aux_type | full | 0.102 | 0.147 | 0.466 | 0.391 |
| mlp_ztype__crl_best_aux_type | focal | 0.102 | 0.000 | 0.565 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.368 | 0.000 | 0.394 |
