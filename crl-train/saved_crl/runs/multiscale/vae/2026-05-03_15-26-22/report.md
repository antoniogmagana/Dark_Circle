# Full diagnostic report — 2026-05-03_15-26-22

## CRL pre-training

- frontend: `multiscale`, d_z=32, d_model=64, n_layers=2

- elapsed: 512.68 min

- total epochs recorded: 100

- **best val_ref_elbo:** 0.192563 (epoch 98) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6768 (epoch 35) → `crl_best_aux_type.pth`


## Phase 2 — probes (selected by max val F1, per head)

Each probe trains both heads jointly with two independent optimizers and saves two checkpoints: the presence ckpt is the epoch with max `val_pres_f1`, the type ckpt is the epoch with max `val_type_f1`. These epochs may differ.

### Presence head

| run | probe | ckpt | best_epoch | val_pres_f1 | val_pres_acc |
|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 9 | 0.8714 | 0.8009 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 1 | 0.8738 | 0.8116 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 31 | 0.8714 | 0.7975 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 1 | 0.8742 | 0.8113 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 48 | 0.8713 | 0.8013 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 1 | 0.8715 | 0.8107 |

### Type head

| run | probe | ckpt | best_epoch | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 37 | 0.4648 | 0.6892 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 32 | 0.5185 | 0.7215 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 2 | 0.5758 | 0.6722 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 34 | 0.5981 | 0.7433 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 1 | 0.5349 | 0.6911 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 16 | 0.6551 | 0.7533 |

## Phase 3 — test evals

Each eval row is from a single head's checkpoint: presence rows come from `downstream_best_pres.pth`, type rows from `downstream_best_type.pth`. Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt, m3nvc, per-vehicle) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

### Presence head — test pres_f1 by split

| run | split | n_windows | pres_f1 |
|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.8761 |
| linear_fullz__crl_best | focal | 32,674 | 0.8600 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.7996 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.9333 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.8488 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.8050 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.9419 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.9082 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.9125 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.8355 |
| linear_fullz__crl_best | iobt | 1,575 | 0.7974 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.7852 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.7752 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.8833 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.8856 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.8822 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.8876 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.8689 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.9021 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8666 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8365 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.7898 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.9264 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.7923 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7449 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9448 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.9083 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.9122 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.8091 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.7291 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.6949 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7016 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.8933 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8837 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8881 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8903 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8557 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8974 |
| linear_ztype__crl_best | full | 91,325 | 0.8766 |
| linear_ztype__crl_best | focal | 32,674 | 0.8616 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.8042 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.9330 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.8526 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.8067 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.9425 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.9077 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.9137 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.8381 |
| linear_ztype__crl_best | iobt | 1,575 | 0.8021 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.7920 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7776 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.8833 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.8855 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8818 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8874 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.8691 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.9022 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.8704 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.8454 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.8023 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.9293 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.8071 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7597 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9446 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.9099 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.9135 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8261 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.7512 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.7187 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7325 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.9007 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8849 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8879 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8887 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8605 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.9000 |
| mlp_ztype__crl_best | full | 91,325 | 0.8786 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8703 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.8287 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.9353 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.8705 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.8135 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.9423 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.9067 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.9225 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.8543 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.8224 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.8198 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7978 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.8758 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.8840 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8793 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8837 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.8714 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.9007 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.8717 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.8490 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.8073 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.9308 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.8129 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7656 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9440 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.9092 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.9150 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8331 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.7520 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.7199 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7336 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.9007 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8852 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8875 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8879 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8623 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.9007 |

### Type head — test type macro_f1 by split

| run | split | n_windows | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.5130 | 0.5130 | 0.6768 |
| linear_fullz__crl_best | focal | 32,674 | 0.3693 | 0.3693 | 0.4209 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.0409 | 0.1636 | 0.0891 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.2331 | 0.9322 | 0.8730 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.1804 | 0.7215 | 0.5644 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.2025 | 0.8099 | 0.6805 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.0963 | 0.3850 | 0.2384 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.0038 | 0.0151 | 0.0076 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.0829 | 0.3315 | 0.1987 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.1099 | 0.4395 | 0.2816 |
| linear_fullz__crl_best | iobt | 1,575 | 0.2133 | 0.4265 | 0.2602 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.0210 | 0.0839 | 0.0438 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.2404 | 0.9615 | 0.9259 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.0034 | 0.0138 | 0.0069 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.3992 | 0.7984 | 0.8246 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.2109 | 0.8437 | 0.7297 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.2298 | 0.9194 | 0.8508 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.2330 | 0.9320 | 0.8727 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.2286 | 0.9143 | 0.8422 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.6068 | 0.6068 | 0.7078 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.4572 | 0.4572 | 0.4742 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0912 | 0.3649 | 0.2232 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.2052 | 0.8210 | 0.6964 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.2198 | 0.8792 | 0.7845 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.1557 | 0.6227 | 0.4521 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.1195 | 0.4780 | 0.3141 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.0727 | 0.2908 | 0.1702 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.1080 | 0.4321 | 0.2756 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.2135 | 0.8542 | 0.7455 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.2831 | 0.5662 | 0.4303 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1151 | 0.4605 | 0.2991 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2429 | 0.9714 | 0.9444 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0325 | 0.1299 | 0.0694 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.4067 | 0.8134 | 0.8402 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2181 | 0.8725 | 0.7738 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.2285 | 0.9141 | 0.8419 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2338 | 0.9353 | 0.8785 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2321 | 0.9282 | 0.8660 |
| linear_ztype__crl_best | full | 91,325 | 0.4544 | 0.4544 | 0.6880 |
| linear_ztype__crl_best | focal | 32,674 | 0.2907 | 0.2907 | 0.4221 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.2391 | 0.9564 | 0.9165 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.1892 | 0.7567 | 0.6087 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.2397 | 0.9589 | 0.9210 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.0605 | 0.2421 | 0.1378 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.0082 | 0.0327 | 0.0166 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.0312 | 0.1248 | 0.0665 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best | iobt | 1,575 | 0.2273 | 0.4545 | 0.2528 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.0244 | 0.0977 | 0.0514 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.2337 | 0.9349 | 0.8778 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.0034 | 0.0138 | 0.0069 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.4062 | 0.8124 | 0.8416 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.2178 | 0.8711 | 0.7717 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.2261 | 0.9044 | 0.8254 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.2372 | 0.9487 | 0.9025 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.2325 | 0.9299 | 0.8690 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.5050 | 0.5050 | 0.7051 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.3545 | 0.3545 | 0.4517 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0000 | 0.0000 | 0.0000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.2273 | 0.9092 | 0.8335 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2115 | 0.8459 | 0.7329 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.2344 | 0.9377 | 0.8827 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0875 | 0.3499 | 0.2120 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.0452 | 0.1807 | 0.0993 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0707 | 0.2828 | 0.1647 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.0427 | 0.1708 | 0.0934 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.2833 | 0.5667 | 0.3857 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.0968 | 0.3873 | 0.2402 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2419 | 0.9675 | 0.9370 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0102 | 0.0408 | 0.0208 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.4105 | 0.8210 | 0.8491 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2207 | 0.8827 | 0.7901 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.2274 | 0.9095 | 0.8340 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2367 | 0.9468 | 0.8990 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2334 | 0.9334 | 0.8751 |
| mlp_ztype__crl_best | full | 91,325 | 0.5482 | 0.5482 | 0.6464 |
| mlp_ztype__crl_best | focal | 32,674 | 0.3604 | 0.3604 | 0.3814 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.1409 | 0.5637 | 0.3925 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.1136 | 0.4543 | 0.2940 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.2332 | 0.9327 | 0.8739 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.1158 | 0.4634 | 0.3015 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.0742 | 0.2970 | 0.1744 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.1758 | 0.7032 | 0.5423 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.0556 | 0.2223 | 0.1251 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.1555 | 0.6222 | 0.4516 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.3079 | 0.6157 | 0.4284 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1110 | 0.4442 | 0.2855 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.2389 | 0.9555 | 0.9148 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.0740 | 0.2959 | 0.1736 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.3931 | 0.7862 | 0.7941 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.2041 | 0.8166 | 0.6901 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.2299 | 0.9198 | 0.8514 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.2252 | 0.9007 | 0.8193 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.2235 | 0.8938 | 0.8080 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.5749 | 0.5749 | 0.7180 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.4112 | 0.4112 | 0.4728 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.0100 | 0.0402 | 0.0205 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.2300 | 0.9201 | 0.8520 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2110 | 0.8441 | 0.7302 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.2190 | 0.8762 | 0.7797 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0773 | 0.3092 | 0.1829 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.0398 | 0.1592 | 0.0865 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0543 | 0.2171 | 0.1218 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.1538 | 0.6154 | 0.4444 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.2899 | 0.5797 | 0.3829 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.0959 | 0.3834 | 0.2372 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2414 | 0.9655 | 0.9333 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0102 | 0.0408 | 0.0208 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.4144 | 0.8288 | 0.8580 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2243 | 0.8974 | 0.8139 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.2253 | 0.9012 | 0.8201 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2387 | 0.9549 | 0.9136 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2352 | 0.9410 | 0.8886 |

## Per-class type F1 on test splits

From the type head only (`downstream_best_type.pth`).

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.282 | 0.391 | 0.769 | 0.610 |
| linear_fullz__crl_best | focal | 0.284 | 0.437 | 0.537 | 0.219 |
| linear_fullz__crl_best | focal__bicycle2 | 0.164 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.932 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.722 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.810 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.385 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.015 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.332 |
| linear_fullz__crl_best | focal__walk2 | 0.440 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.072 | 0.000 | 0.781 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.084 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.962 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.014 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.872 | 0.725 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.844 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.919 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.932 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.914 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.500 | 0.484 | 0.796 | 0.647 |
| linear_fullz__crl_best_aux_type | focal | 0.505 | 0.494 | 0.513 | 0.317 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.365 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.821 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.879 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.623 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.478 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.291 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.432 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.854 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.409 | 0.000 | 0.723 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.461 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.971 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.130 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.885 | 0.742 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.873 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.914 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.935 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.928 | 0.000 |
| linear_ztype__crl_best | full | 0.000 | 0.411 | 0.777 | 0.630 |
| linear_ztype__crl_best | focal | 0.000 | 0.456 | 0.564 | 0.143 |
| linear_ztype__crl_best | focal__bicycle2 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.956 | 0.000 |
| linear_ztype__crl_best | focal__motor2 | 0.000 | 0.757 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.959 | 0.000 |
| linear_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.242 |
| linear_ztype__crl_best | focal__scooter2 | 0.000 | 0.033 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.125 |
| linear_ztype__crl_best | focal__walk2 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.083 | 0.000 | 0.826 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.098 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.935 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.014 | 0.000 | 0.000 |
| linear_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.886 | 0.738 |
| linear_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.871 | 0.000 |
| linear_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.904 |
| linear_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.949 | 0.000 |
| linear_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.930 | 0.000 |
| linear_ztype__crl_best_aux_type | full | 0.099 | 0.483 | 0.797 | 0.642 |
| linear_ztype__crl_best_aux_type | focal | 0.099 | 0.503 | 0.589 | 0.227 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.909 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.846 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.938 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.350 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.181 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.283 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 0.171 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.334 | 0.000 | 0.799 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.387 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.968 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.041 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.892 | 0.750 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.883 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.909 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.947 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.933 | 0.000 |
| mlp_ztype__crl_best | full | 0.361 | 0.470 | 0.762 | 0.600 |
| mlp_ztype__crl_best | focal | 0.407 | 0.475 | 0.401 | 0.158 |
| mlp_ztype__crl_best | focal__bicycle2 | 0.564 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.454 | 0.000 |
| mlp_ztype__crl_best | focal__motor2 | 0.000 | 0.933 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.463 | 0.000 |
| mlp_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.297 |
| mlp_ztype__crl_best | focal__scooter2 | 0.000 | 0.703 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.222 |
| mlp_ztype__crl_best | focal__walk2 | 0.622 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.418 | 0.000 | 0.814 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.444 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.956 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.296 | 0.000 | 0.000 |
| mlp_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.848 | 0.724 |
| mlp_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.817 | 0.000 |
| mlp_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.920 |
| mlp_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.901 | 0.000 |
| mlp_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.894 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.367 | 0.481 | 0.800 | 0.652 |
| mlp_ztype__crl_best_aux_type | focal | 0.367 | 0.502 | 0.571 | 0.205 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 0.040 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.920 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.844 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.876 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.309 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.159 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.217 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 0.615 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.331 | 0.000 | 0.829 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.383 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.966 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.041 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.900 | 0.758 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.897 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.901 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.955 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.941 | 0.000 |
