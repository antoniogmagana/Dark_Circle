# TODO — FFT-conv in MultiScale + STFT-input frontend exploration

Filed 2026-04-27. Two related but distinct items, both deferred until the
ks=159 multiscale run is in and we know whether longer kernels help.

---

## (a) FFT-conv inside `MultiScale1DFrontend`

### When it matters
- **Skip for ks ≤ ~256.** Below that, cuDNN's direct conv is faster than the rfft + complex mul + irfft pipeline. FFT-conv would slow ks=159 down.
- **Worth adding only if we push to ks ≥ 512–800.** That's the regime where FFT-conv beats direct conv (Morlet variants already auto-switch at ks=512 — see `MorletFilterbank._fft_morlet_conv`).

### Why we'd add it
If the ks=159 run validates the "longer kernels help" hypothesis, the next step is ks=799 to actually reach ~20 Hz at SR=16000. At that kernel size, direct conv is the bottleneck and FFT-conv is the standard fix.

### What to do
1. Copy the FFT-conv path from `crl_vehicle/models/frontend.py::MorletFilterbank._fft_morlet_conv` into `MultiScale1DFrontend`. Roughly 30 lines.
2. Auto-switch by ks threshold: `if ks >= 512: fft_conv else: F.conv1d`. Per-branch (one branch can be direct, another FFT, in the same frontend).
3. Differentiability: `torch.fft.rfft` / `irfft` are autograd-clean, so the Conv1D weight `nn.Parameter` flows gradients through the FFT just like the learnable Morlet does today. Slight extra autograd memory at small ks — another reason to gate on threshold.

### Verification when implemented
Run a tiny smoke test: build a `MultiScale1DFrontend` with `kernel_sizes=[39, 799]`, feed it a fixed seed input, assert that switching the FFT path on/off produces outputs within 1e-4 (numerical equivalence). If gradients differ, something is wrong.

### Not worth doing now
The current ks=159 audio default doesn't trigger this. **Only revisit after ks=799 is on the table.**

---

## (b) STFT-input frontend (spectrogram → Conv2D)

### What it would be
Replace raw waveform input with a fixed STFT / mel-spectrogram up front. Conv stack then operates on `(B, freq_bins, time_frames)` instead of `(B, 1, samples)`. Reaching low frequencies becomes a question of STFT bin spacing, not kernel size.

### Why it's interesting
- This is essentially what the Morlet frontend already does (constant-Q wavelet bank). So the "spectrogram + learnable Conv2D" question overlaps heavily with "switch to Morlet."
- A learnable post-stage on top of a fixed STFT is a common pattern in audio (e.g. wav2vec, AST, PANNs) — gives both spectro-temporal inductive bias *and* learning capacity.

### Why it's not a quick win
- Fundamentally different topology. Not a config tweak — frontend, encoder input shape, and decoder reconstruction target all have to change.
- We'd lose the learnable-filterbank advantage that's making `multiscale_v2` our strongest run.
- A hybrid (STFT → small Conv2D) is ~1–2 weeks of focused work, not a one-shot config experiment.

### When to revisit
**After** the ks=159 result. Two cases:

1. **ks=159 helps**: kernel-size matters. Push to ks=799 (with FFT-conv from item a). STFT-input becomes irrelevant — multiscale is doing the right thing, just needed bigger receptive fields.
2. **ks=159 doesn't help**: kernel-size wasn't the bottleneck. *Then* the spectrogram-frontend question becomes worth real time — multiscale's inductive bias may not be the right one, and STFT or Morlet may. But even there, the cleaner first move is to compare `multiscale_v2` head-to-head against `morlet_per_sensor` again on the same training mode, not to invent a new architecture.

### Decision rule
Don't start (b) until both (a) at ks=799 has been tried *and* the morlet vs multiscale head-to-head has been re-checked under disentangled mode.

---

## Crosslink

These items are also relevant to:
- `TODO_false_presence_audit.md` (different concern: validating labels, not architecture)
- The disentangled run still in progress (`disentangled_multiscale_run1`) — its results will inform whether the "kernel size matters" hypothesis is even tested under the right loss landscape.
