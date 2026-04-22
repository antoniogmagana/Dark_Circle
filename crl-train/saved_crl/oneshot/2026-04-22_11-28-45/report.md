# Full diagnostic report — 2026-04-22_11-28-45

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 102.91 min

- total epochs recorded: 70

- **best val_ref_elbo:** 0.54103 (epoch 44) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6797 (epoch 29) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 41 | 0.8679 | 0.6839 | 0.7558 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 14 | 0.8821 | 0.6791 | 0.7403 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 8 | 0.8806 | 0.6876 | 0.7519 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 17 | 0.8813 | 0.6803 | 0.7295 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 47 | 0.8688 | 0.6814 | 0.7534 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 46 | 0.8805 | 0.6790 | 0.7392 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7753 | 0.2131 | 0.2131 | 0.2353 |
| linear_fullz__crl_best | focal | 9,228 | 0.7943 | 0.1281 | 0.2562 | 0.1983 |
| linear_fullz__crl_best | iobt | 3,204 | 0.7151 | 0.1836 | 0.3673 | 0.3539 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.7547 | 0.2149 | 0.2149 | 0.2245 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.7820 | 0.1227 | 0.2453 | 0.1819 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.6611 | 0.1946 | 0.3891 | 0.3608 |
| linear_ztype__crl_best | full | 12,432 | 0.7740 | 0.2200 | 0.2200 | 0.2421 |
| linear_ztype__crl_best | focal | 9,228 | 0.7926 | 0.1305 | 0.2610 | 0.2052 |
| linear_ztype__crl_best | iobt | 3,204 | 0.7149 | 0.1901 | 0.3802 | 0.3603 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.7621 | 0.2356 | 0.2356 | 0.2481 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.7884 | 0.1388 | 0.2776 | 0.2123 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.6729 | 0.2008 | 0.4015 | 0.3627 |
| mlp_ztype__crl_best | full | 12,432 | 0.7931 | 0.2247 | 0.2247 | 0.2383 |
| mlp_ztype__crl_best | focal | 9,228 | 0.8117 | 0.1211 | 0.2422 | 0.1841 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.7349 | 0.2207 | 0.4414 | 0.4115 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.7582 | 0.2321 | 0.2321 | 0.2329 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.7848 | 0.1220 | 0.2441 | 0.1759 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.6671 | 0.2285 | 0.4570 | 0.4150 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.102 | 0.119 | 0.387 | 0.245 |
| linear_fullz__crl_best | focal | 0.106 | 0.000 | 0.407 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.156 | 0.000 | 0.579 |
| linear_fullz__crl_best_aux_type | full | 0.134 | 0.124 | 0.332 | 0.269 |
| linear_fullz__crl_best_aux_type | focal | 0.142 | 0.000 | 0.348 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.168 | 0.000 | 0.611 |
| linear_ztype__crl_best | full | 0.106 | 0.128 | 0.391 | 0.256 |
| linear_ztype__crl_best | focal | 0.110 | 0.000 | 0.412 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.171 | 0.000 | 0.589 |
| linear_ztype__crl_best_aux_type | full | 0.135 | 0.129 | 0.390 | 0.289 |
| linear_ztype__crl_best_aux_type | focal | 0.143 | 0.000 | 0.412 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.175 | 0.000 | 0.628 |
| mlp_ztype__crl_best | full | 0.085 | 0.173 | 0.378 | 0.263 |
| mlp_ztype__crl_best | focal | 0.087 | 0.000 | 0.397 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.286 | 0.000 | 0.597 |
| mlp_ztype__crl_best_aux_type | full | 0.114 | 0.174 | 0.352 | 0.288 |
| mlp_ztype__crl_best_aux_type | focal | 0.120 | 0.000 | 0.368 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.288 | 0.000 | 0.626 |
