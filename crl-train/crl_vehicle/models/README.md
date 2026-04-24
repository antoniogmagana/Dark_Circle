# `crl_vehicle/models/` — neural network components

All nn.Modules that make up `CRLModel`. The model is assembled in `training/trainer.py::CRLModel.__init__` from these pieces based on `config.frontend_type`.

## Files

| File | Contents |
|---|---|
| `frontend.py` | `MultiScale1DFrontend`, `MorletFilterbank`, `LearnableMorletFilterbank` |
| `encoder_decoder.py` | `TemporalEncoder` (Transformer) + `FeatureDecoder` (MLP) |
| `latent.py` | `CausalLatentSpace` — partition of z into causal blocks |
| `intervention.py` | `UnknownInterventionClassifier` + `label_change_target` |
| `heads.py` | `LinearPresenceHead`, `LinearTypeHead`, `MLPTypeHead`, `FullZTypeHead`, `LinearProximityHead` |

---

## `frontend.py` — 6 variants in 3 classes

### `MultiScale1DFrontend`
Three parallel `Conv1D` branches with different kernel sizes (default `[9, 19, 39]`), each followed by GroupNorm + GELU. Outputs are concatenated on the channel axis and projected back to `d_model` by a 1×1 conv. Runs in fp32 to avoid overflow on large audio inputs.

Wired by `frontend_type="multiscale"` (early fusion, per-sensor frontend + shared encoder).

### `MorletFilterbank`
Fixed analytic Morlet wavelets, no learnable parameters. Log-spaced frequencies between `freq_min` and `freq_max`; real and imaginary kernels built at init:

```
scales    = (w0 / 2π) / logspace(freq_min, freq_max, out_channels)
t         = linspace(-ks/2, ks/2, ks) / sample_rate     # SECONDS
kernel_re = (π·s)^(-0.25) · exp(-½(t/s)²) · cos(w0·t/s)
kernel_im = (π·s)^(-0.25) · exp(-½(t/s)²) · sin(w0·t/s)
```

`t` in seconds (not samples) is the fix for the pre-Checkpoint-2 underflow bug where kernels went to zero at SR ≥ 400 with non-default freq ranges.

**Phase output** (`use_phase=True`): forward returns `[log_power, cos_phase, sin_phase]` on the channel axis → 3× out_channels. Phase is computed per bin as `(re/mag, im/mag)` with `mag = sqrt(re² + im² + 1e-8)`. Represented in the unit circle so downstream layers see a differentiable, wrap-free signal.

Wired by `frontend_type="morlet"` (SR-derived heuristic freq range), `"morlet_per_sensor"` (explicit per-sensor ranges from `morlet_per_sensor_params`), and `"morlet_fused"` (same as per_sensor but with `AdaptiveAvgPool1d` → time-concat → shared encoder).

### `LearnableMorletFilterbank(MorletFilterbank)`
Scales become `nn.Parameter` in log space (`log_scales`). `scales = exp(log_scales)` on every forward pass — keeps scales positive without a hard clamp and gives gradients a well-behaved magnitude. Per-filter `w0` becomes an `nn.Parameter` when `learnable_w0=True`.

Initialization exactly matches `MorletFilterbank` at init: `log_scales = log(init_scales)` from the same formula. Epoch-0 output is bit-equivalent within float32 precision. The parent's `kernel_re` / `kernel_im` buffers are **deregistered** in `__init__` since kernels are rebuilt from parameters every forward.

Wired by `frontend_type="morlet_learnable"` (late fusion) and `"morlet_learnable_fused"` (early fusion).

### FFT-based convolution
Both fixed and learnable Morlet variants use FFT conv automatically when `kernel_size ≥ 512`. Below that threshold, cuDNN's direct conv is faster. Implementation in `_fft_morlet_conv`:
- Zero-pad to `n_fft = next_pow2(L + ks - 1)`
- rFFT input + both kernels separately
- Multiply `X · conj(K)` to get cross-correlation (matches `F.conv1d` semantics)
- `torch.roll` the output by `ks//2` to align with direct-conv centering
- Slice to length `L`

Delivers ~3× audio speedup on CPU (ks=4585), larger on GPU. Autograd flows through because `torch.fft.rfft`/`irfft` are differentiable — critical for the learnable variant.

### Per-sensor kernel derivation (coupled to SR and target token count)

For `morlet_per_sensor`, `morlet_fused`, and their learnable variants, `kernel_size` and `pool_stride` are not hardcoded. Per-sensor formulas:

```
pool_stride = window_size // target_tokens
kernel_size = round(2 · receptive_cycles · w0 / (2π · freq_min) · SR)   # odd, ≥ 3
```

`kernel_size` is sized to capture `receptive_cycles` full cycles at the lowest frequency of interest (longest kernel). `pool_stride` normalizes output sequence length to `target_tokens` regardless of SR. This is what makes the frontend **data-agnostic**: change window size or sample rate in config, kernel and stride recompute.

Example per-sensor shapes at defaults:
- Audio (SR=16000, L=16000, freq_min=20, w0=6, cycles=3, target_tokens=32):
  kernel_size = 4585, pool_stride = 500
- Seismic (SR=200, L=200, freq_min=2, w0=6, cycles=3, target_tokens=32):
  kernel_size = 573, pool_stride = 6

Derived values are recorded in `model._morlet_derived_params[sensor]` and persisted in `meta.json` for reproducibility.

---

## `encoder_decoder.py`

### `TemporalEncoder`
Transformer encoder (2 layers, 4 heads, `d_model=64` by default) with norm_first LayerNorm. Input: `(B, C, T)`; input_proj maps C → d_model, transformer does self-attention, mean-pool over T → `(B, d_model)`, then linear `mu_head` + `lv_head` produce `(mu, logvar)` of shape `(B, d_z)`.

`mu` is clamped to `[-10, 10]`, `logvar` to `[-4, 4]` to prevent posterior collapse / explosion. At train time, `z = mu + exp(0.5·logvar) · eps` (reparameterization); at eval time, `z = mu` (deterministic).

Runs the transformer in fp32 — softmax with long sequences and fp16 can NaN.

### `FeatureDecoder`
Simple MLP: `Linear(d_z → d_model) + GELU + Linear(d_model → out_channels·seq_len)` reshaped to `(B, out_channels, seq_len)`. No upsampling/deconv — the decoder reconstructs the frontend's *output* (post-pool tokens), not the raw waveform. Reconstruction loss is MSE in that feature space.

---

## `latent.py` — `CausalLatentSpace`

Partitions the d_z latent into four causal blocks plus a free subspace:

| Block | Dims | Slice |
|---|---|---|
| `pres` | 4 | `z[..., 0:4]` |
| `type` | 6 | `z[..., 4:10]` |
| `prox` | 3 | `z[..., 10:13]` |
| `env` | 6 | `z[..., 13:19]` |
| `free` | d_z − 19 | `z[..., 19:d_z]` |

`D_CAUSAL = 19`. `d_z` must be strictly greater to leave a free/nuisance subspace. Default `d_z=24` gives 5 free dims.

Single method: `split(z) → (z_pres, z_type, z_prox, z_env, z_free)`. Every training mode and every aux head uses these slices to read the right subspace. The `env` block is what feeds `UnknownInterventionClassifier`; `pres`/`type`/`prox` feed their respective aux heads.

---

## `intervention.py`

### `label_change_target(det_t, det_tn, type_t, type_tn) → (B, 2) float`
Binary targets `[pres_changed, type_changed]` between anchor `t` and partner `tn`. Used as ground truth for intervention matching.

### `UnknownInterventionClassifier(z_env_t, z_env_tn) → (B, 2) logits`
MLP that takes concatenated `env` blocks from anchor and partner and predicts the 2-bit change vector. Hidden dim 64, two GELU layers, then Linear(64 → 2). Loss is BCE against `label_change_target`.

Pressures the `env` block to factorize temporal change correctly — if presence changes, the classifier must recover that from the env block's movement alone.

---

## `heads.py` — downstream probe heads

Trained post-CRL on the frozen encoder. Four variants selected by `config.probe_mode`:

| Class | `probe_mode` | Input |
|---|---|---|
| `LinearPresenceHead` | (always) | `z_pres` (D_PRES=4) → `Linear(4, 1)` → presence logit |
| `LinearTypeHead` | `linear_ztype` (default) | `z_type` (D_TYPE=6) → `Linear(6, 4)` → 4-class logits |
| `MLPTypeHead` | `mlp_ztype` | `z_type` → `Linear(6, 32) → ReLU → Linear(32, 4)` |
| `FullZTypeHead` | `linear_fullz` | full `z` (d_z=24) → `Linear(24, 4)` |
| `LinearProximityHead` | (always) | `z_prox` (D_PROX=3) → `Linear(3, 1)` |

Also created during CRL training but not activated: `aux_pres_heads` and `aux_type_heads` (one set per head key — `["fused"]` for early-fusion frontends, sensor names for per-sensor). Aux heads use the same splits but are trained under CRL's loss for direct F1 signal during pretraining.
