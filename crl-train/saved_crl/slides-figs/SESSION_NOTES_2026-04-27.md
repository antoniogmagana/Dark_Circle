# Session notes — 2026-04-27

Snapshot of what's done and what's in flight after this work session.
Companion docs in this dir: `TODO_false_presence_audit.md`,
`TODO_fft_and_stft_frontend.md`, `performance_table.md`, `model_basics.md`.

## Code changes (committed in repo as `b9fb0ef "larger multiscale kernels"`)

- **`crl_vehicle/config.py`**
  - New field: `multiscale_kernel_sizes: dict` (per-sensor kernel-size lists).
  - Default `{"audio": [9, 19, 39, 159]}`. Sensors omitted from the dict fall
    through to `MultiScale1DFrontend`'s built-in `[9, 19, 39]`.
  - Comment block annotates the per-sensor max-viable kernel table.
  - `__post_init__` validates: odd-only, ks ≤ window_size per sensor.

- **`training/trainer.py::_init_multiscale`**
  - Now passes `config.multiscale_kernel_sizes.get(sensor)` to the frontend.
  - Single-line plumbing change. No-op for sensors without an override.

- **`crl_vehicle/models/frontend.py`** — unchanged.

### Caveat for resuming old runs
Pre-existing run `meta.json` files don't have `multiscale_kernel_sizes`.
Loading them through `CRLConfig` instantiates with the new default
`{"audio": [9, 19, 39, 159]}` — meaning **resuming an old run silently
changes the audio frontend**. To pin reproducibility for an old run:
`CRLConfig(multiscale_kernel_sizes={})`.

## Next run command (server)

```bash
cd crl-train && python run_full_diagnostic.py \
  --frontend multiscale --use-id-split \
  --id-root ../data_files/parsed/ \
  --out-dir saved_crl/id_split/multiscale_v3_lowfreq
```

Picks up the new audio kernel ladder automatically.

## Slide deliverables in this directory

| File | Status |
|---|---|
| `fig1_downstream_f1_bars.png` | done — multi vs morlet downstream F1 |
| `fig2_downstream_type_f1_curves.png` | done — type F1 across downstream epochs |
| `fig3_crl_val_ref_elbo.png` | done — β-invariant ELBO per pretraining epoch |
| `fig4_crl_aux_type_f1.png` | done — aux type F1 with best-epoch markers |
| `fig6_architecture.png` | done — frontend swap-point schematic |
| `fig7_frontend_multiscale.png` | done — multiscale internals (boxes) |
| `fig8_frontend_morlet.png` | done — morlet internals (boxes) |
| `fig9_training_flow.png` | done — anchor + partner + dual checkpoint |
| `fig10_morlet_anatomy.png` | done — wavelet = envelope × sinusoid |
| `fig11_multiscale_anatomy.png` | done — receptive-field windows on shared signal |
| `fig12_crl_concept.png` | done — vanilla vs causal representation |
| `fig13_confusion_type_all_runs.png` | done — 3-run type CMs |
| `fig14_confusion_presence_all_runs.png` | done — 3-run presence CMs |
| `model_basics.md` | done — backbone + frontend differences + numbers |
| `performance_table.md` | done — full performance table for 3 completed runs |

Generators (re-runnable, idempotent):
- `_make_figures.py` — figs 1–4
- `_make_diagrams.py` — figs 6–9
- `_make_wavelet_diagram.py` — fig 10
- `_make_multiscale_diagram.py` — fig 11
- `_make_crl_concept.py` — fig 12
- `_make_confusion_all.py` — figs 13–14
- `_make_confusion.py` — older 2-panel version (kept for reference)

All figures: 300 DPI, 16:9 (or 1×N panels), viridis-friendly palette,
presentation-scale fonts (titles 16–18 pt, sub-text 13–15 pt).

## Open threads (deferred work)

### `disentangled_multiscale_run1`
In progress at session end — epoch 82 of 100, last modified 2026-04-27.
Once it finishes, both `_make_confusion_all.py` and `_make_figures.py`
will pick it up automatically if you pass it via `--runs`. No code
change needed to include it in the table or CMs.

### Aux-pres bias-only collapse
We diagnosed this against the **legacy `vae` mode** with the 4-block
causal latent. The user clarified the intent is the disentangled
signal/env split, which is what `disentangled_multiscale_run1` is
running. The collapse hypothesis (intervention-invariance loss
dominating aux_pres) is testable by:
1. Watching `val_aux_pres_f1` in the disentangled run when it finishes.
2. If it's low, run with `--config-overrides-json '{"lambda_interv_inv": 0.0}'`
   to confirm the loss is the cause.
3. If invariance is the cause, rebalance to 0.3 or 0.1.

### `TODO_false_presence_audit.md`
Per-window false-presence analysis. Requires extending `eval.py` to
write per-window predictions. Out of scope for tonight; full plan in
that file.

### `TODO_fft_and_stft_frontend.md`
Two architectural follow-ups:
- (a) FFT-conv inside MultiScale (only relevant if pushing to ks ≥ 512).
- (b) STFT-input frontend (bigger topology change; revisit only if
  ks=159 doesn't help).

Decision rule for both is in that file.

## Known multiscale frontend gaps not addressed today

- Audio reaches ~100 Hz with ks=159; doesn't yet reach 20 Hz (would need
  ks=799 + FFT-conv). Treat ks=159 as the *first* test of the
  longer-kernel hypothesis, not the final form.
- Seismic at SR=100 with `[9, 19, 39]` already covers down to 2.5 Hz;
  no extension needed there.

## What I wouldn't trust without verification next session

- The `multiscale_v3_lowfreq` run (when it lands) needs an updated
  performance_table row. Re-run `_make_confusion_all.py` with
  `--runs ... saved_crl/id_split/multiscale_v3_lowfreq` to add it.
- If you want fig1–4 (training curves) for the new run, re-run
  `_make_figures.py` with the appropriate `--multi-run` argument.
  The script defaults are still pinned to `multiscale_run1`.
