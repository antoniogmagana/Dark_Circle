# `crl_vehicle/data/` — datasets and intervention signals

Parquet-backed sensor datasets for CRL training, plus 7 intervention noise generators used by the intervention-matching loss.

## Files

| File | Public API |
|---|---|
| `dataset.py` | `SensorDataset`, `StratifiedPairDataset`, `collate_single`, `collate_pairs`, `compute_class_weights` |
| `transforms.py` | `remove_dc`, `apply_intervention`, `N_INTERVENTIONS=7` |

## `SensorDataset`

Reads parquet files under `<data_dir>/` matching the naming pattern `{dataset}_{sensor}_{vehicle}_{rs_node}.parquet`. One row per **window** (not per sample); windows are non-overlapping contiguous slices of the raw amplitude stream, sized per `config.modality_cfg(sensor).window_size`.

### On-disk schema
Each parquet file has two columns:
- `amplitude` (float32) — raw signal samples
- `present` (bool) — per-sample presence flag (vehicle in scene)

Windowing: per-window label is the **majority vote** of the `present` column (strictly > 50% → presence=1). Trailing samples that don't fill a complete window are discarded.

### Supported datasets
`iobt`, `focal`, `m3nvc`. Vehicle name → class mapping lives in `config.DATASET_VEHICLE_MAP`. Multi-vehicle compound stems (`m3nvc` only, detected by an underscore in the vehicle name) are excluded with `LABEL_MULTI` so they don't contaminate training.

### Cache
Windows are materialized once into a per-config cache dir (`SensorDataset(..., cache_dir=...)`) and memory-mapped thereafter. Cache hash keys include the window size, sample rate, and sensor — changing any of these invalidates the cache automatically.

### `__getitem__` returns
```python
{
    "x_<sensor>":          (C, W) float tensor,
    "<sensor>_avail":      bool (always True for this row — availability is
                                 per-sensor in paired datasets; see below),
    "detection_label":     {0, 1} int (presence),
    "vehicle_type":        int class or LABEL_BACKGROUND (-1) / LABEL_MULTI (-2),
    "dataset":             str name,
    ...
}
```

### Key helper: `compute_class_weights(ds) → (pres_pos_weight, type_class_weights)`

Returns the `BCEWithLogitsLoss(pos_weight=...)` value for presence and the `CrossEntropyLoss(weight=...)` tensor for type. Used by `Trainer.train_downstream()` to counteract class imbalance.

## `StratifiedPairDataset`

Wraps a `SensorDataset` to produce **anchor + partner** pairs for CRL training. Each anchor gets N partners drawn from four strata:

| Stratum constant | Meaning |
|---|---|
| `STRATUM_CONSEC` | Consecutive window from the same `(dataset, vehicle, rs_node)` segment |
| `STRATUM_SAME_TYPE` | Different segment, same vehicle type |
| `STRATUM_DIFF_TYPE` | Different vehicle type |
| `STRATUM_CROSS_DS` | Different dataset entirely |

Partner counts per stratum come from `config.n_partners_same_type`, `n_partners_diff_type`, `n_partners_cross_ds`; `STRATUM_CONSEC` is always 1 partner (if available).

### Why stratified partners?

Two training modes use them differently:

- **VAE mode** uses the first partner (`p0`) for CITRIS-style intervention matching: "which latent block changed between anchor and partner?" Label-change targets come from `label_change_target(det_t, det_p0, type_t, type_p0)` in `models/intervention.py`.
- **Contrastive mode** uses the stratum tag to decide positives vs negatives: `CONSEC` and `SAME_TYPE` are positives; `DIFF_TYPE` and `CROSS_DS` are negatives.

A single partner sampling strategy serves both modes — no mode-specific data plumbing.

## Collate functions

- `collate_single(batch)` — stacks one-per-sample dicts for downstream/eval. Presence/type as `(B,)` tensors.
- `collate_pairs(batch)` — stacks anchor + N partners for CRL training. Produces `x_<sensor>_t`, `x_<sensor>_p0`, ..., `x_<sensor>_pN`, plus `detection_label_t`, `detection_label_p0`, `partner_stratum_p0`, etc.

Both handle audio-only or seismic-only availability via the `*_avail` bool keys; training modes mask loss accumulation accordingly.

## `transforms.py` — intervention noise generators

`apply_intervention(x, intervention_id, sample_rate)` adds one of 7 noise types at 20% RMS of the signal:

1. white
2. brown (1/f²)
3. pink (1/f)
4. green (300-800 Hz band-limited)
5. low-frequency sinusoid (2-12 Hz random)
6. high-frequency chirp (1 kHz → 8 kHz linear sweep)
7. bird-chirp-like band-limited noise (2-4 kHz Gaussian envelope)

Currently used only for held-out intervention eval, not online during training. `N_INTERVENTIONS = 7` is exposed so callers don't hardcode it.

`remove_dc(x)` subtracts the per-channel mean. Applied once on raw windows before passing to the frontend; prevents DC offsets from biasing the Morlet kernel's low-frequency end.
