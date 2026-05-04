# Full diagnostic report — 2026-05-03_05-02-44

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 300.76 min

- total epochs recorded: 51

- **best val_ref_elbo:** 0.552606 (epoch 25) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6071 (epoch 4) → `crl_best_aux_type.pth`


## Phase 2 — probes (selected by max val F1, per head)

Each probe trains both heads jointly with two independent optimizers and saves two checkpoints: the presence ckpt is the epoch with max `val_pres_f1`, the type ckpt is the epoch with max `val_type_f1`. These epochs may differ.

### Presence head

| run | probe | ckpt | best_epoch | val_pres_f1 | val_pres_acc |
|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 9 | 0.8625 | 0.8027 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 23 | 0.8489 | 0.7846 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 31 | 0.8670 | 0.8064 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 17 | 0.8485 | 0.7843 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 48 | 0.8650 | 0.8050 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 18 | 0.8479 | 0.7838 |

### Type head

| run | probe | ckpt | best_epoch | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 3 | 0.6391 | 0.7233 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 36 | 0.6120 | 0.6921 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 35 | 0.6419 | 0.7413 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 4 | 0.6112 | 0.6995 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 1 | 0.6428 | 0.7364 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 20 | 0.6121 | 0.6911 |

## Phase 3 — test evals

Each eval row is from a single head's checkpoint: presence rows come from `downstream_best_pres.pth`, type rows from `downstream_best_type.pth`. Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt, m3nvc, per-vehicle) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

### Presence head — test pres_f1 by split

| run | split | n_windows | pres_f1 |
|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.8551 |
| linear_fullz__crl_best | focal | 32,674 | 0.8431 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.7693 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.9154 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.7764 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.7765 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.9404 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.9020 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.9032 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.8524 |
| linear_fullz__crl_best | iobt | 1,575 | 0.6083 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.4297 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.7500 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.9073 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.8656 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.8683 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.8806 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.8294 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.8789 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.8357 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.8115 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6896 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.8814 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.7490 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7371 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9233 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8802 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8698 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.8460 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.5146 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.3468 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5736 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.8993 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8529 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8532 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8616 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8316 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8625 |
| linear_ztype__crl_best | full | 91,325 | 0.8511 |
| linear_ztype__crl_best | focal | 32,674 | 0.8374 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7583 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.9112 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.7703 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7685 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.9383 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.8990 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.8973 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.8484 |
| linear_ztype__crl_best | iobt | 1,575 | 0.5921 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.4061 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7330 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.9073 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.8626 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8656 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8788 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.8260 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.8743 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.8370 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.8132 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6923 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.8831 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7529 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7389 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9243 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8813 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8730 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8461 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.5185 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.3515 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5810 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.8993 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8540 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8539 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8623 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8331 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8639 |
| mlp_ztype__crl_best | full | 91,325 | 0.8581 |
| mlp_ztype__crl_best | focal | 32,674 | 0.8474 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7786 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.9177 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.7874 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7813 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.9411 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.9030 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.9070 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.8563 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.6166 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.4518 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7461 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.9013 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.8680 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8705 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8816 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.8332 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.8818 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.8364 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.8122 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6904 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.8824 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7501 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7380 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9238 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8806 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8710 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8458 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.5175 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.3495 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5810 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.8993 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8536 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8538 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8621 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8324 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8636 |

### Type head — test type macro_f1 by split

| run | split | n_windows | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 91,325 | 0.5869 | 0.5869 | 0.7039 |
| linear_fullz__crl_best | focal | 32,674 | 0.4409 | 0.4409 | 0.4615 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.0711 | 0.2845 | 0.1658 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.1442 | 0.5768 | 0.4052 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.2190 | 0.8761 | 0.7795 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.1664 | 0.6657 | 0.4990 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.1060 | 0.4239 | 0.2689 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.1459 | 0.5836 | 0.4121 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.1014 | 0.4056 | 0.2544 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.2099 | 0.8394 | 0.7232 |
| linear_fullz__crl_best | iobt | 1,575 | 0.3142 | 0.6284 | 0.4703 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1351 | 0.5402 | 0.3701 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.2389 | 0.9555 | 0.9148 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.0443 | 0.1772 | 0.0972 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.3950 | 0.7901 | 0.8399 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.2372 | 0.9490 | 0.9029 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.1956 | 0.7825 | 0.6427 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.2389 | 0.9557 | 0.9152 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.2397 | 0.9587 | 0.9206 |
| linear_fullz__crl_best_aux_type | full | 91,325 | 0.5607 | 0.5607 | 0.6471 |
| linear_fullz__crl_best_aux_type | focal | 32,674 | 0.3803 | 0.3803 | 0.3975 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1047 | 0.4190 | 0.2650 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.0528 | 0.2113 | 0.1181 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.2308 | 0.9234 | 0.8577 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.0925 | 0.3700 | 0.2270 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0903 | 0.3613 | 0.2205 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1881 | 0.7524 | 0.6031 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.1132 | 0.4527 | 0.2926 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.2153 | 0.8613 | 0.7564 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.2848 | 0.5696 | 0.4898 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1374 | 0.5498 | 0.3792 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2472 | 0.9888 | 0.9778 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0384 | 0.1538 | 0.0833 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.3673 | 0.7347 | 0.7851 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2238 | 0.8953 | 0.8104 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.1931 | 0.7723 | 0.6291 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2319 | 0.9277 | 0.8651 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2302 | 0.9209 | 0.8533 |
| linear_ztype__crl_best | full | 91,325 | 0.5833 | 0.5833 | 0.6982 |
| linear_ztype__crl_best | focal | 32,674 | 0.4554 | 0.4554 | 0.4633 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.0633 | 0.2531 | 0.1449 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.1147 | 0.4587 | 0.2976 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.2158 | 0.8634 | 0.7596 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.1671 | 0.6684 | 0.5020 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.1498 | 0.5993 | 0.4278 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.1220 | 0.4880 | 0.3227 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.1462 | 0.5850 | 0.4134 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.2049 | 0.8198 | 0.6947 |
| linear_ztype__crl_best | iobt | 1,575 | 0.2880 | 0.5759 | 0.4638 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1273 | 0.5090 | 0.3414 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.2462 | 0.9850 | 0.9704 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.0355 | 0.1419 | 0.0764 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.3935 | 0.7870 | 0.8301 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.2314 | 0.9257 | 0.8617 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.2048 | 0.8191 | 0.6936 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.2347 | 0.9389 | 0.8848 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.2361 | 0.9446 | 0.8950 |
| linear_ztype__crl_best_aux_type | full | 91,325 | 0.5607 | 0.5607 | 0.6485 |
| linear_ztype__crl_best_aux_type | focal | 32,674 | 0.3807 | 0.3807 | 0.3976 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1040 | 0.4161 | 0.2627 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.0530 | 0.2120 | 0.1185 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2308 | 0.9231 | 0.8572 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.0929 | 0.3715 | 0.2281 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0906 | 0.3626 | 0.2214 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1875 | 0.7499 | 0.5998 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.1146 | 0.4583 | 0.2973 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.2151 | 0.8604 | 0.7549 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.2848 | 0.5696 | 0.4898 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1374 | 0.5498 | 0.3792 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2472 | 0.9888 | 0.9778 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0384 | 0.1538 | 0.0833 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.3678 | 0.7356 | 0.7872 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2246 | 0.8983 | 0.8154 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.1918 | 0.7673 | 0.6224 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2326 | 0.9304 | 0.8698 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2311 | 0.9246 | 0.8597 |
| mlp_ztype__crl_best | full | 91,325 | 0.5784 | 0.5784 | 0.7056 |
| mlp_ztype__crl_best | focal | 32,674 | 0.4239 | 0.4239 | 0.4651 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.0539 | 0.2157 | 0.1209 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.2010 | 0.8040 | 0.6722 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.2101 | 0.8405 | 0.7248 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.1881 | 0.7523 | 0.6029 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.0856 | 0.3422 | 0.2064 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.0975 | 0.3902 | 0.2424 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.0558 | 0.2231 | 0.1255 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.1982 | 0.7928 | 0.6567 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.3178 | 0.6356 | 0.4210 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.1221 | 0.4886 | 0.3233 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.2300 | 0.9200 | 0.8519 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.0294 | 0.1176 | 0.0625 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.3954 | 0.7909 | 0.8419 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.2381 | 0.9522 | 0.9087 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.1945 | 0.7781 | 0.6368 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.2396 | 0.9585 | 0.9203 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.2402 | 0.9608 | 0.9245 |
| mlp_ztype__crl_best_aux_type | full | 91,325 | 0.5642 | 0.5642 | 0.6590 |
| mlp_ztype__crl_best_aux_type | focal | 32,674 | 0.3760 | 0.3760 | 0.4058 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.1194 | 0.4777 | 0.3138 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.0567 | 0.2267 | 0.1278 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.2324 | 0.9294 | 0.8681 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.1042 | 0.4170 | 0.2634 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.0650 | 0.2601 | 0.1495 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.1986 | 0.7946 | 0.6592 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.0745 | 0.2980 | 0.1751 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.2187 | 0.8746 | 0.7772 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.3341 | 0.6682 | 0.5195 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.1505 | 0.6019 | 0.4305 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.2448 | 0.9792 | 0.9593 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.0472 | 0.1887 | 0.1042 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.3693 | 0.7386 | 0.7984 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.2311 | 0.9245 | 0.8596 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.1821 | 0.7284 | 0.5728 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.2368 | 0.9473 | 0.8998 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.2351 | 0.9402 | 0.8871 |

## Per-class type F1 on test splits

From the type head only (`downstream_best_type.pth`).

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.469 | 0.470 | 0.818 | 0.590 |
| linear_fullz__crl_best | focal | 0.474 | 0.469 | 0.528 | 0.293 |
| linear_fullz__crl_best | focal__bicycle2 | 0.284 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.577 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.876 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.666 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.424 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.584 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.406 |
| linear_fullz__crl_best | focal__walk2 | 0.839 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.484 | 0.000 | 0.773 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.540 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.956 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.177 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.893 | 0.687 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.949 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.782 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.956 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.959 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.473 | 0.476 | 0.768 | 0.526 |
| linear_fullz__crl_best_aux_type | focal | 0.480 | 0.477 | 0.309 | 0.256 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.419 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.211 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.923 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.370 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.361 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.752 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.453 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.861 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.490 | 0.000 | 0.649 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.550 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.989 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.154 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.852 | 0.618 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.895 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.772 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.928 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.921 | 0.000 |
| linear_ztype__crl_best | full | 0.461 | 0.467 | 0.816 | 0.590 |
| linear_ztype__crl_best | focal | 0.465 | 0.468 | 0.540 | 0.348 |
| linear_ztype__crl_best | focal__bicycle2 | 0.253 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.459 | 0.000 |
| linear_ztype__crl_best | focal__motor2 | 0.000 | 0.863 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.668 | 0.000 |
| linear_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.599 |
| linear_ztype__crl_best | focal__scooter2 | 0.000 | 0.488 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.585 |
| linear_ztype__crl_best | focal__walk2 | 0.820 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.452 | 0.000 | 0.700 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.509 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.985 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.142 | 0.000 | 0.000 |
| linear_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.883 | 0.691 |
| linear_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.926 | 0.000 |
| linear_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.819 |
| linear_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.939 | 0.000 |
| linear_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.945 | 0.000 |
| linear_ztype__crl_best_aux_type | full | 0.472 | 0.475 | 0.770 | 0.525 |
| linear_ztype__crl_best_aux_type | focal | 0.479 | 0.476 | 0.310 | 0.258 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 0.416 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.212 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.923 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.371 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.363 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.750 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.458 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 0.860 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.490 | 0.000 | 0.649 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.550 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.989 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.154 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.854 | 0.618 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.898 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.767 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.930 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.925 | 0.000 |
| mlp_ztype__crl_best | full | 0.452 | 0.459 | 0.804 | 0.599 |
| mlp_ztype__crl_best | focal | 0.455 | 0.463 | 0.535 | 0.243 |
| mlp_ztype__crl_best | focal__bicycle2 | 0.216 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.804 | 0.000 |
| mlp_ztype__crl_best | focal__motor2 | 0.000 | 0.841 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.752 | 0.000 |
| mlp_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.342 |
| mlp_ztype__crl_best | focal__scooter2 | 0.000 | 0.390 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.223 |
| mlp_ztype__crl_best | focal__walk2 | 0.793 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.432 | 0.000 | 0.839 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.489 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.920 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.118 | 0.000 | 0.000 |
| mlp_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.894 | 0.687 |
| mlp_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.952 | 0.000 |
| mlp_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.778 |
| mlp_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.959 | 0.000 |
| mlp_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.961 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.479 | 0.469 | 0.782 | 0.526 |
| mlp_ztype__crl_best_aux_type | focal | 0.488 | 0.466 | 0.341 | 0.209 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 0.478 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.227 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.929 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.417 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.260 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.795 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.298 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 0.875 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.538 | 0.000 | 0.798 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.602 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.979 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.189 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.865 | 0.612 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.924 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.728 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.947 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.940 | 0.000 |
