# TODO — false-presence audit (deferred from 2026-04-27)

## Goal

Inspect the false presence predictions across completed id_split runs to see whether the underlying labels are actually wrong. Hypothesis: if multiple models with different inductive biases agree on a "wrong" prediction, the label is the more likely error.

## Why it isn't a 30-minute task

`eval_report.json` only stores aggregate counts (`tp/fp/fn/tn` plus the 4×4 type confusion). **Per-window predictions are not persisted.** Need to add per-window dump first.

## Approach (when we pick this up)

1. **Extend `eval.py`** — write per-window CSV alongside `eval_report.json` during inference.
   Columns: `dataset`, `vehicle`, `rs_node`, `window_start_time`, `pred_logit`, `pred_prob`, `pred_label`, `true_label`. The loader already runs `shuffle=False`, so window identity is recoverable from `dataset._index` (see `per_vehicle_confusion.py` for the same pattern).
   Output path: `<run>/eval/<probe>__<ckpt>/full/predictions.csv`.

2. **Re-run eval on the relevant runs** — `multiscale_v2` and `morlet_per_sensor_phase_run1_diag` (both n=91,325, same split — these can be intersected). Skip `multiscale_run1_diag` (different `use_id_split`, different test set).

3. **Write `_false_presence_audit.py`** in this dir. Two outputs:

   - `false_presence_intersection.csv`: rows where *all included models* agree on a false prediction. This is the high-signal set — labels are most likely to be the error here.
   - `fig15_false_presence_grid.png`: waveform grid (audio + seismic) for ~12–20 sampled rows from the intersection, captioned `(true label, all-model prediction, file path)`.

## Open decisions for when we resume

- **Which models to include in the intersection?** Default to the two ID-split runs above. Add `disentangled_multiscale_run1` once it finishes if you want a 3-way intersection.
- **Sample size for the visual grid.** 12 rows fit cleanly on one slide; more than 20 stops being readable.
- **Sampling strategy.** Random vs. by-confidence. Highest-confidence false predictions are the most surprising and best for "is this a labeling error" investigation.
- **Server vs local.** Plotting waveforms requires raw `parsed/` files which live on the server. Probably a server-side script that writes the PNG, then `scp` the result.

## Not in scope today

The CRL training/eval/figure pipeline does not need to change. This is purely a post-hoc analysis on already-trained models.
