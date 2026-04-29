# Full diagnostic report — 2026-04-25_09-41-39

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 480.68 min

- total epochs recorded: 68

- **best val_ref_elbo:** 0.143751 (epoch 42) → `crl_best.pth`

- **best val_aux_type_f1:** 0.5057 (epoch 23) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_signal__crl_best | linear_signal | crl_best.pth | 42 | 0.8251 | 0.4729 | 0.6093 |
| linear_signal__crl_best_aux_type | linear_signal | crl_best_aux_type.pth | 37 | 0.8175 | 0.4999 | 0.6258 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 16 | 0.8382 | 0.4714 | 0.6076 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 40 | 0.8273 | 0.4995 | 0.6270 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 21,436 | 0.8094 | 0.3657 | 0.3657 | 0.4778 |
| linear_fullz__crl_best | focal | 14,534 | 0.8178 | 0.2409 | 0.2409 | 0.2679 |
| linear_fullz__crl_best | iobt | 936 | 0.8072 | 0.3287 | 0.6575 | 0.6634 |
| linear_fullz__crl_best_aux_type | full | 21,436 | 0.8155 | 0.3471 | 0.3471 | 0.4385 |
| linear_fullz__crl_best_aux_type | focal | 14,534 | 0.8153 | 0.1966 | 0.1966 | 0.2073 |
| linear_fullz__crl_best_aux_type | iobt | 936 | 0.7424 | 0.4079 | 0.8157 | 0.8035 |
| linear_signal__crl_best | full | 21,436 | 0.8007 | 0.3588 | 0.3588 | 0.4753 |
| linear_signal__crl_best | focal | 14,534 | 0.8119 | 0.2373 | 0.2373 | 0.2664 |
| linear_signal__crl_best | iobt | 936 | 0.8070 | 0.3136 | 0.6272 | 0.6409 |
| linear_signal__crl_best_aux_type | full | 21,436 | 0.8089 | 0.3521 | 0.3521 | 0.4392 |
| linear_signal__crl_best_aux_type | focal | 14,534 | 0.8110 | 0.2029 | 0.2029 | 0.2109 |
| linear_signal__crl_best_aux_type | iobt | 936 | 0.7271 | 0.4065 | 0.8131 | 0.8035 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.021 | 0.286 | 0.658 | 0.497 |
| linear_fullz__crl_best | focal | 0.021 | 0.255 | 0.399 | 0.288 |
| linear_fullz__crl_best | iobt | 0.000 | 0.558 | 0.000 | 0.757 |
| linear_fullz__crl_best_aux_type | full | 0.037 | 0.252 | 0.598 | 0.501 |
| linear_fullz__crl_best_aux_type | focal | 0.037 | 0.176 | 0.279 | 0.294 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.785 | 0.000 | 0.846 |
| linear_signal__crl_best | full | 0.021 | 0.262 | 0.657 | 0.495 |
| linear_signal__crl_best | focal | 0.021 | 0.233 | 0.399 | 0.297 |
| linear_signal__crl_best | iobt | 0.000 | 0.512 | 0.000 | 0.742 |
| linear_signal__crl_best_aux_type | full | 0.035 | 0.274 | 0.598 | 0.502 |
| linear_signal__crl_best_aux_type | focal | 0.035 | 0.204 | 0.271 | 0.302 |
| linear_signal__crl_best_aux_type | iobt | 0.000 | 0.785 | 0.000 | 0.841 |
