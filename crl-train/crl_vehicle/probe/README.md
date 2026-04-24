# `crl_vehicle/probe/` — inference-time log-prior shift

Post-training correction for probe classifiers when the eval split has a different class distribution than training. Used by `eval.py --recalibrate` and downstream probe reporting.

## Files

| File | Contents |
|---|---|
| `recalibration.py` | `apply_binary_log_prior_shift`, `apply_multiclass_log_prior_shift`, plus prior computation helpers |

## Why this exists

Probes in this codebase train with **class-balanced loss** (`BCEWithLogitsLoss(pos_weight=...)` for presence, `CrossEntropyLoss(weight=...)` for type). That makes the effective training prior approximately uniform over classes.

When you evaluate on a split with a different class distribution (say, 90% "light" vehicles in focal, 20% in m3nvc), the probe's argmax biases toward whichever class dominates the split. The log-prior shift corrects this:

```
logit_adjusted[c] = logit[c] - log(p_train[c]) + log(p_split[c])
```

This yields the classification the representation would give **under the split's true prior**. It doesn't change the representation or the trained weights — just adjusts the decision threshold post-hoc.

## API

### `apply_binary_log_prior_shift(logits, p_split, p_train=0.5) → logits`
For presence (BCE). `p_train=0.5` matches uniform-via-pos_weight training.

### `apply_multiclass_log_prior_shift(logits, p_split, p_train=None) → logits`
For type (K-way cross entropy). `p_train` defaults to uniform `1/K`.

### `compute_binary_prior(labels) → float` and `compute_multiclass_prior(labels, n_classes) → Tensor`
Empirical priors from label tensors. Used to derive `p_split` from the eval set.

## Use only for diagnostics with known priors

The shift is an **oracle correction**: it requires knowing the eval split's true class distribution. That's fine for diagnostic reporting (where you have labels) but inappropriate for deployment where target priors are unknown.

Convention: report columns that have been shifted carry the `_target_calibrated` suffix so they can't be confused with uncalibrated deployment numbers. See `eval.py` for the exact metric names.

## What to read alongside

Your prior work (memory `project_crl_feedback_server_workflow.md` / probe-recalibration notes) flagged a degeneracy: some runs had `presence_calibrated.f1 = 0.83` but `balanced_accuracy = 0.5` + `MCC = 0.0`. That's the classifier predicting positive always — calibrated F1 still looks fine because the shift makes positive match the split prior, but the model isn't actually discriminating. **Always read `balanced_accuracy` and `mcc` next to `_target_calibrated` F1.** The analysis scripts (`compare_runs.py`) include both columns for this reason.
