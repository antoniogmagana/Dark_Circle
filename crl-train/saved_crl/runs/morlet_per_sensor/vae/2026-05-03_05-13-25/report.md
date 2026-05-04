# Full diagnostic report — 2026-05-03_05-13-25

## CRL pre-training

- frontend: `morlet_per_sensor`, d_z=24, d_model=64, n_layers=2

- elapsed: 390.66 min

- total epochs recorded: 66

- **best val_ref_elbo:** 802.942783 (epoch 40) → `crl_best.pth`

- **best val_aux_type_f1:** 0.4591 (epoch 6) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 42 | 0.8037 | 0.4718 | 0.6424 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 10 | 0.8303 | 0.4531 | 0.6153 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 9 | 0.4857 | 0.4702 | 0.6546 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 14 | 0.8307 | 0.4525 | 0.6095 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 23 | 0.4860 | 0.3655 | 0.3851 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 9 | 0.8304 | 0.4585 | 0.6253 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.4918 | 0.3485 | 0.3485 | 0.3655 |
| linear_fullz__crl_best | focal | 29,980 | 0.5637 | 0.2804 | 0.2804 | 0.2683 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.5343 | 0.0660 | 0.2640 | 0.1521 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.6273 | 0.0033 | 0.0132 | 0.0067 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.5451 | 0.1028 | 0.4113 | 0.2589 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.5213 | 0.0759 | 0.3036 | 0.1790 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.6274 | 0.1736 | 0.6943 | 0.5317 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.6044 | 0.1583 | 0.6331 | 0.4632 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.5808 | 0.1737 | 0.6948 | 0.5323 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.5659 | 0.1500 | 0.5999 | 0.4284 |
| linear_fullz__crl_best | iobt | 1,575 | 0.5527 | 0.1345 | 0.2691 | 0.2551 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.5956 | 0.0345 | 0.1378 | 0.0740 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.5327 | 0.2222 | 0.8889 | 0.8000 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.3743 | 0.0309 | 0.1238 | 0.0660 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.4477 | 0.2137 | 0.4274 | 0.4158 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.4102 | 0.1240 | 0.4959 | 0.3297 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.4669 | 0.2082 | 0.8327 | 0.7133 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.3951 | 0.1284 | 0.5137 | 0.3456 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.5051 | 0.0986 | 0.3945 | 0.2457 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.8338 | 0.4399 | 0.4399 | 0.5877 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.8046 | 0.3391 | 0.3391 | 0.4286 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.7282 | 0.0569 | 0.2276 | 0.1284 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.9020 | 0.1691 | 0.6766 | 0.5113 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.7890 | 0.1146 | 0.4584 | 0.2973 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7478 | 0.2038 | 0.8153 | 0.6881 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.8955 | 0.0333 | 0.1330 | 0.0712 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8536 | 0.1410 | 0.5641 | 0.3928 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8142 | 0.0232 | 0.0927 | 0.0486 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.8345 | 0.1543 | 0.6172 | 0.4463 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.7955 | 0.1291 | 0.2581 | 0.1868 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.8451 | 0.0781 | 0.3123 | 0.1850 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7467 | 0.0859 | 0.3436 | 0.2074 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6437 | 0.0676 | 0.2703 | 0.1562 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8494 | 0.2996 | 0.5992 | 0.6753 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8484 | 0.2164 | 0.8655 | 0.7630 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8664 | 0.1424 | 0.5695 | 0.3982 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8290 | 0.2305 | 0.9221 | 0.8554 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8497 | 0.2096 | 0.8383 | 0.7216 |
| linear_ztype__crl_best | full | 88,631 | 0.8142 | 0.4571 | 0.4571 | 0.6111 |
| linear_ztype__crl_best | focal | 29,980 | 0.8276 | 0.3520 | 0.3520 | 0.4507 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7907 | 0.0251 | 0.1005 | 0.0529 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.9122 | 0.1880 | 0.7519 | 0.6024 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.8633 | 0.1392 | 0.5567 | 0.3857 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7454 | 0.2114 | 0.8454 | 0.7322 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.9129 | 0.0326 | 0.1305 | 0.0698 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.8671 | 0.1341 | 0.5366 | 0.3667 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.8716 | 0.0185 | 0.0741 | 0.0385 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.8671 | 0.1560 | 0.6241 | 0.4536 |
| linear_ztype__crl_best | iobt | 1,575 | 0.8137 | 0.1787 | 0.3573 | 0.2179 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.8725 | 0.0447 | 0.1788 | 0.0982 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.7187 | 0.1621 | 0.6483 | 0.4796 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.7678 | 0.1087 | 0.4348 | 0.2778 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.8067 | 0.3101 | 0.6202 | 0.6992 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.7843 | 0.2203 | 0.8811 | 0.7875 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8184 | 0.1466 | 0.5864 | 0.4148 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.7772 | 0.2309 | 0.9235 | 0.8579 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.8409 | 0.2179 | 0.8715 | 0.7723 |
| linear_ztype__crl_best_aux_type | full | 88,631 | 0.8336 | 0.4379 | 0.4379 | 0.5827 |
| linear_ztype__crl_best_aux_type | focal | 29,980 | 0.8050 | 0.3369 | 0.3369 | 0.4266 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.7293 | 0.0447 | 0.1789 | 0.0982 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.9028 | 0.1727 | 0.6908 | 0.5276 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7895 | 0.1187 | 0.4746 | 0.3111 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7479 | 0.2045 | 0.8180 | 0.6921 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.8957 | 0.0358 | 0.1432 | 0.0771 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8538 | 0.1323 | 0.5294 | 0.3600 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8152 | 0.0271 | 0.1085 | 0.0573 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8352 | 0.1540 | 0.6162 | 0.4453 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.7956 | 0.1368 | 0.2737 | 0.2082 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.8454 | 0.0868 | 0.3471 | 0.2100 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7467 | 0.0840 | 0.3359 | 0.2019 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6437 | 0.0874 | 0.3496 | 0.2118 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8490 | 0.2994 | 0.5987 | 0.6682 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8479 | 0.2111 | 0.8443 | 0.7305 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8662 | 0.1484 | 0.5936 | 0.4220 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8287 | 0.2281 | 0.9123 | 0.8387 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8492 | 0.2084 | 0.8335 | 0.7145 |
| mlp_ztype__crl_best | full | 88,631 | 0.4915 | 0.4575 | 0.4575 | 0.6268 |
| mlp_ztype__crl_best | focal | 29,980 | 0.5635 | 0.3310 | 0.3310 | 0.4432 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.5333 | 0.0180 | 0.0721 | 0.0374 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.6273 | 0.1899 | 0.7597 | 0.6125 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.5446 | 0.1455 | 0.5819 | 0.4103 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.5211 | 0.2087 | 0.8350 | 0.7167 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.6274 | 0.0124 | 0.0495 | 0.0254 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.6045 | 0.1385 | 0.5540 | 0.3831 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.5805 | 0.0047 | 0.0187 | 0.0094 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.5660 | 0.1531 | 0.6125 | 0.4414 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.5523 | 0.2043 | 0.4086 | 0.2235 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.5949 | 0.0533 | 0.2132 | 0.1193 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.5327 | 0.1516 | 0.6065 | 0.4352 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.3743 | 0.1170 | 0.4681 | 0.3056 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.4473 | 0.3160 | 0.6320 | 0.7264 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.4100 | 0.2314 | 0.9255 | 0.8613 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.4665 | 0.1343 | 0.5371 | 0.3672 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.3946 | 0.2347 | 0.9389 | 0.8848 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.5051 | 0.2276 | 0.9102 | 0.8351 |
| mlp_ztype__crl_best_aux_type | full | 88,631 | 0.8334 | 0.4386 | 0.4386 | 0.5792 |
| mlp_ztype__crl_best_aux_type | focal | 29,980 | 0.8071 | 0.3291 | 0.3291 | 0.4164 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.7337 | 0.0436 | 0.1744 | 0.0955 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.9044 | 0.1696 | 0.6786 | 0.5135 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7948 | 0.1288 | 0.5150 | 0.3468 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7483 | 0.1963 | 0.7851 | 0.6462 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.8970 | 0.0283 | 0.1131 | 0.0599 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8550 | 0.1368 | 0.5470 | 0.3764 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8186 | 0.0217 | 0.0867 | 0.0453 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8385 | 0.1565 | 0.6261 | 0.4557 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.7955 | 0.1390 | 0.2780 | 0.2407 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.8446 | 0.1090 | 0.4359 | 0.2787 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.7483 | 0.0552 | 0.2208 | 0.1241 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.6437 | 0.1108 | 0.4432 | 0.2847 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8476 | 0.2999 | 0.5998 | 0.6672 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8461 | 0.2121 | 0.8485 | 0.7368 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8648 | 0.1477 | 0.5907 | 0.4191 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8281 | 0.2259 | 0.9035 | 0.8240 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8478 | 0.2094 | 0.8377 | 0.7208 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.379 | 0.264 | 0.395 | 0.356 |
| linear_fullz__crl_best | focal | 0.402 | 0.332 | 0.219 | 0.168 |
| linear_fullz__crl_best | focal__bicycle2 | 0.264 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.013 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.411 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.304 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.694 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.633 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.695 |
| linear_fullz__crl_best | focal__walk2 | 0.600 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.128 | 0.000 | 0.411 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.138 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.889 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.124 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.438 | 0.417 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.496 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.833 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.514 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.395 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.342 | 0.324 | 0.714 | 0.380 |
| linear_fullz__crl_best_aux_type | focal | 0.379 | 0.337 | 0.535 | 0.105 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.228 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.677 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.458 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.815 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.133 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.564 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.093 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.617 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.282 | 0.000 | 0.235 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.312 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.344 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.270 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.780 | 0.419 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.866 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.570 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.922 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.838 | 0.000 |
| linear_ztype__crl_best | full | 0.362 | 0.324 | 0.733 | 0.411 |
| linear_ztype__crl_best | focal | 0.379 | 0.349 | 0.565 | 0.115 |
| linear_ztype__crl_best | focal__bicycle2 | 0.101 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.752 | 0.000 |
| linear_ztype__crl_best | focal__motor2 | 0.000 | 0.557 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.845 | 0.000 |
| linear_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.131 |
| linear_ztype__crl_best | focal__scooter2 | 0.000 | 0.537 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.074 |
| linear_ztype__crl_best | focal__walk2 | 0.624 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.230 | 0.000 | 0.485 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.179 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.648 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.435 | 0.000 | 0.000 |
| linear_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.797 | 0.443 |
| linear_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.881 | 0.000 |
| linear_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.586 |
| linear_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.923 | 0.000 |
| linear_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.872 | 0.000 |
| linear_ztype__crl_best_aux_type | full | 0.328 | 0.328 | 0.708 | 0.388 |
| linear_ztype__crl_best_aux_type | focal | 0.362 | 0.335 | 0.539 | 0.112 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 0.179 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.691 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.475 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.818 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.143 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.529 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.108 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 0.616 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.321 | 0.000 | 0.226 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.347 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.336 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.350 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.771 | 0.426 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.844 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.594 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.912 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.834 | 0.000 |
| mlp_ztype__crl_best | full | 0.356 | 0.319 | 0.750 | 0.404 |
| mlp_ztype__crl_best | focal | 0.371 | 0.347 | 0.557 | 0.048 |
| mlp_ztype__crl_best | focal__bicycle2 | 0.072 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.760 | 0.000 |
| mlp_ztype__crl_best | focal__motor2 | 0.000 | 0.582 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.835 | 0.000 |
| mlp_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.050 |
| mlp_ztype__crl_best | focal__scooter2 | 0.000 | 0.554 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.019 |
| mlp_ztype__crl_best | focal__walk2 | 0.613 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.263 | 0.000 | 0.554 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.213 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.607 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.468 | 0.000 | 0.000 |
| mlp_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.822 | 0.442 |
| mlp_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.925 | 0.000 |
| mlp_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.537 |
| mlp_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.939 | 0.000 |
| mlp_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.910 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.326 | 0.336 | 0.704 | 0.389 |
| mlp_ztype__crl_best_aux_type | focal | 0.363 | 0.341 | 0.520 | 0.092 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 0.174 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.679 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.515 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.785 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.113 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.547 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.087 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 0.626 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.398 | 0.000 | 0.158 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.436 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.221 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.443 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.771 | 0.428 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.849 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.591 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.903 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.838 | 0.000 |
