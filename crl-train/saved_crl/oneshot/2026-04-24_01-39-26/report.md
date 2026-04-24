# Full diagnostic report — 2026-04-24_01-39-26

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 124.71 min

- total epochs recorded: 77

- **best val_ref_elbo:** 2.309018 (epoch 51) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4245 (epoch 18) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 28 | 0.7762 | 0.5045 | 0.5958 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 44 | 0.7500 | 0.5308 | 0.5911 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 42 | 0.7863 | 0.4618 | 0.5551 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 37 | 0.7620 | 0.4875 | 0.5490 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 33 | 0.7918 | 0.5018 | 0.5827 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 37 | 0.7589 | 0.5325 | 0.5962 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 12,432 | 0.7101 | 0.2492 | 0.2492 | 0.2635 |
| linear_fullz__crl_best | focal | 9,228 | 0.7074 | 0.1841 | 0.3682 | 0.2563 |
| linear_fullz__crl_best | iobt | 3,204 | 0.7176 | 0.1933 | 0.3866 | 0.2866 |
| linear_fullz__crl_best_aux_type | full | 12,432 | 0.6045 | 0.2391 | 0.2391 | 0.2467 |
| linear_fullz__crl_best_aux_type | focal | 9,228 | 0.5964 | 0.1618 | 0.3236 | 0.2282 |
| linear_fullz__crl_best_aux_type | iobt | 3,204 | 0.6279 | 0.2049 | 0.4099 | 0.3058 |
| linear_ztype__crl_best | full | 12,432 | 0.6863 | 0.2664 | 0.2664 | 0.2917 |
| linear_ztype__crl_best | focal | 9,228 | 0.6822 | 0.2061 | 0.4123 | 0.3028 |
| linear_ztype__crl_best | iobt | 3,204 | 0.6978 | 0.1802 | 0.3603 | 0.2563 |
| linear_ztype__crl_best_aux_type | full | 12,432 | 0.5831 | 0.2258 | 0.2258 | 0.2334 |
| linear_ztype__crl_best_aux_type | focal | 9,228 | 0.5727 | 0.1540 | 0.3080 | 0.2126 |
| linear_ztype__crl_best_aux_type | iobt | 3,204 | 0.6129 | 0.2004 | 0.4009 | 0.2999 |
| mlp_ztype__crl_best | full | 12,432 | 0.7024 | 0.2500 | 0.2500 | 0.2234 |
| mlp_ztype__crl_best | focal | 9,228 | 0.6995 | 0.1334 | 0.2668 | 0.2062 |
| mlp_ztype__crl_best | iobt | 3,204 | 0.7106 | 0.2004 | 0.4007 | 0.2785 |
| mlp_ztype__crl_best_aux_type | full | 12,432 | 0.6152 | 0.2203 | 0.2203 | 0.1760 |
| mlp_ztype__crl_best_aux_type | focal | 9,228 | 0.6088 | 0.1003 | 0.2006 | 0.1475 |
| mlp_ztype__crl_best_aux_type | iobt | 3,204 | 0.6338 | 0.1989 | 0.3978 | 0.2671 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.313 | 0.178 | 0.351 | 0.154 |
| linear_fullz__crl_best | focal | 0.348 | 0.000 | 0.389 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.339 | 0.000 | 0.434 |
| linear_fullz__crl_best_aux_type | full | 0.286 | 0.123 | 0.301 | 0.246 |
| linear_fullz__crl_best_aux_type | focal | 0.320 | 0.000 | 0.327 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.279 | 0.000 | 0.541 |
| linear_ztype__crl_best | full | 0.345 | 0.161 | 0.397 | 0.163 |
| linear_ztype__crl_best | focal | 0.387 | 0.000 | 0.438 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.290 | 0.000 | 0.431 |
| linear_ztype__crl_best_aux_type | full | 0.274 | 0.123 | 0.284 | 0.222 |
| linear_ztype__crl_best_aux_type | focal | 0.308 | 0.000 | 0.308 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.275 | 0.000 | 0.527 |
| mlp_ztype__crl_best | full | 0.326 | 0.138 | 0.142 | 0.395 |
| mlp_ztype__crl_best | focal | 0.378 | 0.000 | 0.156 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.404 | 0.000 | 0.397 |
| mlp_ztype__crl_best_aux_type | full | 0.192 | 0.119 | 0.164 | 0.407 |
| mlp_ztype__crl_best_aux_type | focal | 0.225 | 0.000 | 0.177 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.355 | 0.000 | 0.440 |
