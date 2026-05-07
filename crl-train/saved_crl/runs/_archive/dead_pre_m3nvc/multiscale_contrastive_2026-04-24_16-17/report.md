# Full diagnostic report — 2026-04-24_16-17-43

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 36.63 min

- total epochs recorded: 35

- **best val_ref_elbo:** inf (epoch -1) → `crl_best.pth`

- **best val_aux_type_f1:** 0.0 (epoch 0) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 1 | 0.4161 | 0.3920 | 0.4549 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 43 | 0.7382 | 0.4483 | 0.5069 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 2 | 0.5848 | 0.4647 | 0.5252 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.5356 | 0.4033 | 0.4033 | 0.4350 |
| linear_fullz__crl_best | focal | 14,534 | 0.4190 | 0.2168 | 0.2168 | 0.2211 |
| linear_fullz__crl_best | iobt | 936 | 0.6109 | 0.3845 | 0.7690 | 0.7118 |
| linear_ztype__crl_best | full | 21,436 | 0.3584 | 0.3076 | 0.3076 | 0.3277 |
| linear_ztype__crl_best | focal | 14,534 | 0.2253 | 0.1708 | 0.1708 | 0.1658 |
| linear_ztype__crl_best | iobt | 936 | 0.4848 | 0.2579 | 0.5159 | 0.4138 |
| mlp_ztype__crl_best | full | 21,436 | 0.6711 | 0.3247 | 0.3247 | 0.3519 |
| mlp_ztype__crl_best | focal | 14,534 | 0.5574 | 0.2039 | 0.2039 | 0.1748 |
| mlp_ztype__crl_best | iobt | 936 | 0.7031 | 0.1006 | 0.2013 | 0.1836 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.107 | 0.441 | 0.615 | 0.451 |
| linear_fullz__crl_best | focal | 0.107 | 0.395 | 0.049 | 0.317 |
| linear_fullz__crl_best | iobt | 0.000 | 0.847 | 0.000 | 0.691 |
| linear_ztype__crl_best | full | 0.057 | 0.371 | 0.479 | 0.323 |
| linear_ztype__crl_best | focal | 0.058 | 0.346 | 0.011 | 0.269 |
| linear_ztype__crl_best | iobt | 0.000 | 0.634 | 0.000 | 0.398 |
| mlp_ztype__crl_best | full | 0.091 | 0.329 | 0.481 | 0.399 |
| mlp_ztype__crl_best | focal | 0.092 | 0.355 | 0.060 | 0.308 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.041 | 0.000 | 0.362 |
