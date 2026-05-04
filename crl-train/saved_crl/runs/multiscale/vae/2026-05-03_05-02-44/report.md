# Full diagnostic report — 2026-05-03_05-02-44

## CRL pre-training

- frontend: `multiscale`, d_z=24, d_model=64, n_layers=2

- elapsed: 300.76 min

- total epochs recorded: 51

- **best val_ref_elbo:** 0.552606 (epoch 25) → `crl_best.pth`

- **best val_aux_type_f1:** 0.6071 (epoch 4) → `crl_best_aux_type.pth`


## Phase 2 — probes (best val by val_loss)

| run | probe | ckpt | best_epoch | val_pres_f1 | val_type_f1 | val_type_acc |
|---|---|---|---|---|---|---|
| linear_ztype__crl_best | linear_ztype | crl_best.pth | 34 | 0.8512 | 0.6539 | 0.7343 |
| linear_ztype__crl_best_aux_type | linear_ztype | crl_best_aux_type.pth | 12 | 0.8445 | 0.6298 | 0.6997 |
| mlp_ztype__crl_best | mlp_ztype | crl_best.pth | 15 | 0.8484 | 0.6592 | 0.7600 |
| mlp_ztype__crl_best_aux_type | mlp_ztype | crl_best_aux_type.pth | 4 | 0.8458 | 0.6347 | 0.7142 |
| linear_fullz__crl_best | linear_fullz | crl_best.pth | 11 | 0.8604 | 0.6561 | 0.7390 |
| linear_fullz__crl_best_aux_type | linear_fullz | crl_best_aux_type.pth | 30 | 0.8444 | 0.6288 | 0.6970 |

## Phase 3 — test evals

Macro F1 is averaged over all 4 classes; filtered splits (focal, iobt) exclude some classes entirely, so `macro_f1_support_only` restricts the average to classes with support > 0 in that split and is the fair cross-split comparison.

| run | split | n_windows | pres_f1 | type_macro_f1 | type_macro_f1_support_only | type_acc |
|---|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 88,631 | 0.8463 | 0.5988 | 0.5988 | 0.7094 |
| linear_fullz__crl_best | focal | 29,980 | 0.8265 | 0.4420 | 0.4420 | 0.4934 |
| linear_fullz__crl_best | focal__bicycle2 | 3,911 | 0.7505 | 0.0578 | 0.2310 | 0.1306 |
| linear_fullz__crl_best | focal__forester2 | 2,975 | 0.9084 | 0.1810 | 0.7240 | 0.5673 |
| linear_fullz__crl_best | focal__motor2 | 2,734 | 0.7596 | 0.2167 | 0.8669 | 0.7650 |
| linear_fullz__crl_best | focal__mustang0528 | 10,643 | 0.7633 | 0.1806 | 0.7223 | 0.5653 |
| linear_fullz__crl_best | focal__pickup2 | 2,548 | 0.9378 | 0.0868 | 0.3473 | 0.2102 |
| linear_fullz__crl_best | focal__scooter2 | 2,762 | 0.8972 | 0.1256 | 0.5025 | 0.3356 |
| linear_fullz__crl_best | focal__tesla2 | 2,694 | 0.8916 | 0.0709 | 0.2835 | 0.1652 |
| linear_fullz__crl_best | focal__walk2 | 4,407 | 0.8436 | 0.2014 | 0.8057 | 0.6747 |
| linear_fullz__crl_best | iobt | 1,575 | 0.5792 | 0.3199 | 0.6399 | 0.4545 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.3885 | 0.1293 | 0.5174 | 0.3489 |
| linear_fullz__crl_best | iobt__silverado0315pm | 490 | 0.7202 | 0.2384 | 0.9535 | 0.9111 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 190 | 0.9037 | 0.0384 | 0.1538 | 0.0833 |
| linear_fullz__crl_best | m3nvc | 57,076 | 0.8601 | 0.3907 | 0.7815 | 0.8213 |
| linear_fullz__crl_best | m3nvc__cx30 | 14,020 | 0.8623 | 0.2276 | 0.9104 | 0.8355 |
| linear_fullz__crl_best | m3nvc__gle350 | 15,317 | 0.8778 | 0.2096 | 0.8384 | 0.7217 |
| linear_fullz__crl_best | m3nvc__miata | 13,441 | 0.8221 | 0.2316 | 0.9266 | 0.8633 |
| linear_fullz__crl_best | m3nvc__mustang | 14,298 | 0.8718 | 0.2333 | 0.9333 | 0.8750 |
| linear_fullz__crl_best_aux_type | full | 88,631 | 0.8283 | 0.5816 | 0.5816 | 0.6611 |
| linear_fullz__crl_best_aux_type | focal | 29,980 | 0.7970 | 0.3924 | 0.3924 | 0.4397 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6745 | 0.1136 | 0.4545 | 0.2941 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 2,975 | 0.8761 | 0.0641 | 0.2566 | 0.1472 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 2,734 | 0.7309 | 0.2324 | 0.9294 | 0.8681 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7268 | 0.1183 | 0.4734 | 0.3101 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9186 | 0.0720 | 0.2881 | 0.1683 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8762 | 0.1953 | 0.7813 | 0.6412 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8633 | 0.0848 | 0.3393 | 0.2043 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 4,407 | 0.8414 | 0.2159 | 0.8637 | 0.7601 |
| linear_fullz__crl_best_aux_type | iobt | 1,575 | 0.4850 | 0.3277 | 0.6553 | 0.5167 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.3019 | 0.1479 | 0.5915 | 0.4199 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5464 | 0.2467 | 0.9869 | 0.9741 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.8986 | 0.0472 | 0.1887 | 0.1042 |
| linear_fullz__crl_best_aux_type | m3nvc | 57,076 | 0.8478 | 0.3652 | 0.7305 | 0.7728 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8478 | 0.2175 | 0.8702 | 0.7702 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8590 | 0.2021 | 0.8083 | 0.6783 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8246 | 0.2283 | 0.9131 | 0.8401 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8562 | 0.2243 | 0.8973 | 0.8138 |
| linear_ztype__crl_best | full | 88,631 | 0.8344 | 0.5975 | 0.5975 | 0.7065 |
| linear_ztype__crl_best | focal | 29,980 | 0.8103 | 0.4461 | 0.4461 | 0.4934 |
| linear_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7176 | 0.0576 | 0.2304 | 0.1302 |
| linear_ztype__crl_best | focal__forester2 | 2,975 | 0.8974 | 0.1766 | 0.7063 | 0.5460 |
| linear_ztype__crl_best | focal__motor2 | 2,734 | 0.7227 | 0.2170 | 0.8680 | 0.7668 |
| linear_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7473 | 0.1799 | 0.7198 | 0.5622 |
| linear_ztype__crl_best | focal__pickup2 | 2,548 | 0.9336 | 0.0975 | 0.3899 | 0.2421 |
| linear_ztype__crl_best | focal__scooter2 | 2,762 | 0.8900 | 0.1258 | 0.5030 | 0.3360 |
| linear_ztype__crl_best | focal__tesla2 | 2,694 | 0.8817 | 0.0815 | 0.3262 | 0.1949 |
| linear_ztype__crl_best | focal__walk2 | 4,407 | 0.8288 | 0.2014 | 0.8055 | 0.6744 |
| linear_ztype__crl_best | iobt | 1,575 | 0.5399 | 0.3122 | 0.6243 | 0.4554 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.3424 | 0.1285 | 0.5140 | 0.3459 |
| linear_ztype__crl_best | iobt__silverado0315pm | 490 | 0.6715 | 0.2399 | 0.9595 | 0.9222 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.8973 | 0.0384 | 0.1538 | 0.0833 |
| linear_ztype__crl_best | m3nvc | 57,076 | 0.8505 | 0.3893 | 0.7786 | 0.8168 |
| linear_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8539 | 0.2258 | 0.9033 | 0.8236 |
| linear_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8744 | 0.2116 | 0.8463 | 0.7336 |
| linear_ztype__crl_best | m3nvc__miata | 13,441 | 0.8072 | 0.2301 | 0.9203 | 0.8523 |
| linear_ztype__crl_best | m3nvc__mustang | 14,298 | 0.8583 | 0.2321 | 0.9283 | 0.8662 |
| linear_ztype__crl_best_aux_type | full | 88,631 | 0.8285 | 0.5811 | 0.5811 | 0.6613 |
| linear_ztype__crl_best_aux_type | focal | 29,980 | 0.7974 | 0.3872 | 0.3872 | 0.4336 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6755 | 0.1142 | 0.4568 | 0.2960 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.8763 | 0.0603 | 0.2411 | 0.1371 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7319 | 0.2320 | 0.9281 | 0.8658 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7273 | 0.1128 | 0.4513 | 0.2914 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9184 | 0.0736 | 0.2943 | 0.1725 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8762 | 0.1943 | 0.7771 | 0.6355 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8636 | 0.0876 | 0.3503 | 0.2124 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8416 | 0.2167 | 0.8668 | 0.7649 |
| linear_ztype__crl_best_aux_type | iobt | 1,575 | 0.4860 | 0.3149 | 0.6299 | 0.5102 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.3019 | 0.1452 | 0.5809 | 0.4094 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5464 | 0.2467 | 0.9869 | 0.9741 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.9024 | 0.0472 | 0.1887 | 0.1042 |
| linear_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8480 | 0.3661 | 0.7322 | 0.7763 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8483 | 0.2191 | 0.8763 | 0.7798 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8591 | 0.2000 | 0.8001 | 0.6668 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8250 | 0.2295 | 0.9180 | 0.8485 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8563 | 0.2257 | 0.9029 | 0.8230 |
| mlp_ztype__crl_best | full | 88,631 | 0.8307 | 0.5948 | 0.5948 | 0.7224 |
| mlp_ztype__crl_best | focal | 29,980 | 0.8055 | 0.4240 | 0.4240 | 0.4963 |
| mlp_ztype__crl_best | focal__bicycle2 | 3,911 | 0.7082 | 0.0541 | 0.2163 | 0.1213 |
| mlp_ztype__crl_best | focal__forester2 | 2,975 | 0.8943 | 0.2173 | 0.8691 | 0.7685 |
| mlp_ztype__crl_best | focal__motor2 | 2,734 | 0.7142 | 0.2049 | 0.8197 | 0.6945 |
| mlp_ztype__crl_best | focal__mustang0528 | 10,643 | 0.7430 | 0.1900 | 0.7601 | 0.6131 |
| mlp_ztype__crl_best | focal__pickup2 | 2,548 | 0.9307 | 0.0570 | 0.2282 | 0.1288 |
| mlp_ztype__crl_best | focal__scooter2 | 2,762 | 0.8880 | 0.0774 | 0.3094 | 0.1830 |
| mlp_ztype__crl_best | focal__tesla2 | 2,694 | 0.8753 | 0.0384 | 0.1534 | 0.0831 |
| mlp_ztype__crl_best | focal__walk2 | 4,407 | 0.8235 | 0.1981 | 0.7926 | 0.6564 |
| mlp_ztype__crl_best | iobt | 1,575 | 0.5303 | 0.3275 | 0.6550 | 0.4191 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 895 | 0.3266 | 0.1173 | 0.4694 | 0.3066 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 490 | 0.6634 | 0.2348 | 0.9391 | 0.8852 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 190 | 0.8973 | 0.0294 | 0.1176 | 0.0625 |
| mlp_ztype__crl_best | m3nvc | 57,076 | 0.8474 | 0.3955 | 0.7910 | 0.8403 |
| mlp_ztype__crl_best | m3nvc__cx30 | 14,020 | 0.8516 | 0.2368 | 0.9472 | 0.8996 |
| mlp_ztype__crl_best | m3nvc__gle350 | 15,317 | 0.8730 | 0.1968 | 0.7871 | 0.6489 |
| mlp_ztype__crl_best | m3nvc__miata | 13,441 | 0.8025 | 0.2389 | 0.9555 | 0.9148 |
| mlp_ztype__crl_best | m3nvc__mustang | 14,298 | 0.8537 | 0.2394 | 0.9578 | 0.9190 |
| mlp_ztype__crl_best_aux_type | full | 88,631 | 0.8303 | 0.5885 | 0.5885 | 0.6772 |
| mlp_ztype__crl_best_aux_type | focal | 29,980 | 0.8000 | 0.3881 | 0.3881 | 0.4453 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 3,911 | 0.6792 | 0.1180 | 0.4719 | 0.3088 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 2,975 | 0.8768 | 0.0745 | 0.2979 | 0.1750 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 2,734 | 0.7384 | 0.2335 | 0.9340 | 0.8762 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 10,643 | 0.7302 | 0.1179 | 0.4718 | 0.3087 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 2,548 | 0.9200 | 0.0518 | 0.2073 | 0.1157 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 2,762 | 0.8781 | 0.2026 | 0.8103 | 0.6811 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 2,694 | 0.8657 | 0.0526 | 0.2103 | 0.1175 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 4,407 | 0.8436 | 0.2174 | 0.8696 | 0.7692 |
| mlp_ztype__crl_best_aux_type | iobt | 1,575 | 0.4936 | 0.3529 | 0.7059 | 0.5223 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 895 | 0.3100 | 0.1523 | 0.6092 | 0.4381 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 490 | 0.5602 | 0.2434 | 0.9734 | 0.9481 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 190 | 0.9024 | 0.0500 | 0.2000 | 0.1111 |
| mlp_ztype__crl_best_aux_type | m3nvc | 57,076 | 0.8494 | 0.3711 | 0.7422 | 0.7943 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 14,020 | 0.8494 | 0.2270 | 0.9081 | 0.8317 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 15,317 | 0.8598 | 0.1908 | 0.7634 | 0.6174 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 13,441 | 0.8267 | 0.2348 | 0.9392 | 0.8854 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 14,298 | 0.8583 | 0.2316 | 0.9265 | 0.8631 |

## Per-class type F1 on test splits

| run | split | pedestrian_f1 | light_f1 | medium_f1 | heavy_f1 |
|---|---|---|---|---|---|
| linear_fullz__crl_best | full | 0.471 | 0.491 | 0.800 | 0.634 |
| linear_fullz__crl_best | focal | 0.474 | 0.494 | 0.557 | 0.242 |
| linear_fullz__crl_best | focal__bicycle2 | 0.231 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__forester2 | 0.000 | 0.000 | 0.724 | 0.000 |
| linear_fullz__crl_best | focal__motor2 | 0.000 | 0.867 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.722 | 0.000 |
| linear_fullz__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.347 |
| linear_fullz__crl_best | focal__scooter2 | 0.000 | 0.502 | 0.000 | 0.000 |
| linear_fullz__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.283 |
| linear_fullz__crl_best | focal__walk2 | 0.806 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt | 0.000 | 0.461 | 0.000 | 0.819 |
| linear_fullz__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.517 | 0.000 | 0.000 |
| linear_fullz__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.954 |
| linear_fullz__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.154 | 0.000 | 0.000 |
| linear_fullz__crl_best | m3nvc | 0.000 | 0.000 | 0.875 | 0.688 |
| linear_fullz__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.910 | 0.000 |
| linear_fullz__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.838 |
| linear_fullz__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.927 | 0.000 |
| linear_fullz__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.933 | 0.000 |
| linear_fullz__crl_best_aux_type | full | 0.497 | 0.502 | 0.763 | 0.564 |
| linear_fullz__crl_best_aux_type | focal | 0.505 | 0.502 | 0.391 | 0.172 |
| linear_fullz__crl_best_aux_type | focal__bicycle2 | 0.455 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.257 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__motor2 | 0.000 | 0.929 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.473 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.288 |
| linear_fullz__crl_best_aux_type | focal__scooter2 | 0.000 | 0.781 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.339 |
| linear_fullz__crl_best_aux_type | focal__walk2 | 0.864 | 0.000 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt | 0.000 | 0.530 | 0.000 | 0.780 |
| linear_fullz__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.592 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.987 |
| linear_fullz__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.189 | 0.000 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.839 | 0.622 |
| linear_fullz__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.870 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.808 |
| linear_fullz__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.913 | 0.000 |
| linear_fullz__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.897 | 0.000 |
| linear_ztype__crl_best | full | 0.470 | 0.492 | 0.798 | 0.630 |
| linear_ztype__crl_best | focal | 0.474 | 0.495 | 0.560 | 0.255 |
| linear_ztype__crl_best | focal__bicycle2 | 0.230 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.706 | 0.000 |
| linear_ztype__crl_best | focal__motor2 | 0.000 | 0.868 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.720 | 0.000 |
| linear_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.390 |
| linear_ztype__crl_best | focal__scooter2 | 0.000 | 0.503 | 0.000 | 0.000 |
| linear_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.326 |
| linear_ztype__crl_best | focal__walk2 | 0.805 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt | 0.000 | 0.458 | 0.000 | 0.790 |
| linear_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.514 | 0.000 | 0.000 |
| linear_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.960 |
| linear_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.154 | 0.000 | 0.000 |
| linear_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.871 | 0.686 |
| linear_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.903 | 0.000 |
| linear_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.846 |
| linear_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.920 | 0.000 |
| linear_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.928 | 0.000 |
| linear_ztype__crl_best_aux_type | full | 0.499 | 0.502 | 0.765 | 0.559 |
| linear_ztype__crl_best_aux_type | focal | 0.506 | 0.503 | 0.373 | 0.167 |
| linear_ztype__crl_best_aux_type | focal__bicycle2 | 0.457 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.241 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.928 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.451 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.294 |
| linear_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.777 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.350 |
| linear_ztype__crl_best_aux_type | focal__walk2 | 0.867 | 0.000 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt | 0.000 | 0.521 | 0.000 | 0.739 |
| linear_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.581 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.987 |
| linear_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.189 | 0.000 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.842 | 0.622 |
| linear_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.876 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.800 |
| linear_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.918 | 0.000 |
| linear_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.903 | 0.000 |
| mlp_ztype__crl_best | full | 0.465 | 0.469 | 0.808 | 0.637 |
| mlp_ztype__crl_best | focal | 0.468 | 0.476 | 0.563 | 0.188 |
| mlp_ztype__crl_best | focal__bicycle2 | 0.216 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__forester2 | 0.000 | 0.000 | 0.869 | 0.000 |
| mlp_ztype__crl_best | focal__motor2 | 0.000 | 0.820 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__mustang0528 | 0.000 | 0.000 | 0.760 | 0.000 |
| mlp_ztype__crl_best | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.228 |
| mlp_ztype__crl_best | focal__scooter2 | 0.000 | 0.309 | 0.000 | 0.000 |
| mlp_ztype__crl_best | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.153 |
| mlp_ztype__crl_best | focal__walk2 | 0.793 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt | 0.000 | 0.415 | 0.000 | 0.895 |
| mlp_ztype__crl_best | iobt__polaris0235pm_nolineofsig | 0.000 | 0.469 | 0.000 | 0.000 |
| mlp_ztype__crl_best | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.939 |
| mlp_ztype__crl_best | iobt__warhog_nolineofsight | 0.000 | 0.118 | 0.000 | 0.000 |
| mlp_ztype__crl_best | m3nvc | 0.000 | 0.000 | 0.893 | 0.689 |
| mlp_ztype__crl_best | m3nvc__cx30 | 0.000 | 0.000 | 0.947 | 0.000 |
| mlp_ztype__crl_best | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.787 |
| mlp_ztype__crl_best | m3nvc__miata | 0.000 | 0.000 | 0.956 | 0.000 |
| mlp_ztype__crl_best | m3nvc__mustang | 0.000 | 0.000 | 0.958 | 0.000 |
| mlp_ztype__crl_best_aux_type | full | 0.499 | 0.499 | 0.781 | 0.575 |
| mlp_ztype__crl_best_aux_type | focal | 0.508 | 0.498 | 0.392 | 0.155 |
| mlp_ztype__crl_best_aux_type | focal__bicycle2 | 0.472 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__forester2 | 0.000 | 0.000 | 0.298 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__motor2 | 0.000 | 0.934 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__mustang0528 | 0.000 | 0.000 | 0.472 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__pickup2 | 0.000 | 0.000 | 0.000 | 0.207 |
| mlp_ztype__crl_best_aux_type | focal__scooter2 | 0.000 | 0.810 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | focal__tesla2 | 0.000 | 0.000 | 0.000 | 0.210 |
| mlp_ztype__crl_best_aux_type | focal__walk2 | 0.870 | 0.000 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt | 0.000 | 0.545 | 0.000 | 0.866 |
| mlp_ztype__crl_best_aux_type | iobt__polaris0235pm_nolineofsig | 0.000 | 0.609 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | iobt__silverado0315pm | 0.000 | 0.000 | 0.000 | 0.973 |
| mlp_ztype__crl_best_aux_type | iobt__warhog_nolineofsight | 0.000 | 0.200 | 0.000 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc | 0.000 | 0.000 | 0.860 | 0.625 |
| mlp_ztype__crl_best_aux_type | m3nvc__cx30 | 0.000 | 0.000 | 0.908 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__gle350 | 0.000 | 0.000 | 0.000 | 0.763 |
| mlp_ztype__crl_best_aux_type | m3nvc__miata | 0.000 | 0.000 | 0.939 | 0.000 |
| mlp_ztype__crl_best_aux_type | m3nvc__mustang | 0.000 | 0.000 | 0.926 | 0.000 |
