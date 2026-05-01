from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScale1DFrontend(nn.Module):
    """Parallel multi-scale Conv1D branches with GroupNorm + GELU, merged by 1x1 conv.
    Runs in fp32 to avoid overflow on large-stride audio inputs.

    Per-branch stride: when `strides` is given, each branch uses its own stride
    (e.g. ks // 3 to hop at ~1/3 of the kernel's receptive field). When None,
    falls back to the legacy uniform `stride` for every branch.

    Branch alignment: when `target_tokens` is set, each branch ends with
    AdaptiveAvgPool1d(target_tokens), so concat happens on already-aligned
    tensors regardless of stride. When None, no per-branch pool is added —
    callers using a uniform stride get aligned outputs by construction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int] | None = None,
        stride: int = 4,
        strides: list[int] | None = None,
        target_tokens: int | None = None,
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [9, 19, 39]
        if strides is None:
            strides = [stride] * len(kernel_sizes)
        elif len(strides) != len(kernel_sizes):
            raise ValueError(
                f"strides length {len(strides)} must match kernel_sizes "
                f"length {len(kernel_sizes)}"
            )
        self.branches = nn.ModuleList()
        for ks, s in zip(kernel_sizes, strides, strict=False):
            layers: list[nn.Module] = [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=ks,
                    stride=s,
                    padding=ks // 2,
                    bias=False,
                ),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
            ]
            if target_tokens is not None:
                layers.append(nn.AdaptiveAvgPool1d(target_tokens))
            self.branches.append(nn.Sequential(*layers))
        self.proj = nn.Conv1d(
            len(kernel_sizes) * out_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        feats = [branch(x) for branch in self.branches]
        return self.proj(torch.cat(feats, dim=1))


class MorletFilterbank(nn.Module):
    """Fixed Morlet wavelet filterbank — no learnable parameters.

    If freq_min / freq_max are omitted, falls back to sample-rate-based
    defaults: freq_min = 2 Hz (SR ≤ 200) or 20 Hz (SR > 200); freq_max =
    SR/4. Explicit freq_min/freq_max override the heuristic — use them
    to give audio and seismic sensor-appropriate band allocation.

    If use_phase=True, forward returns [log_power, cos_phase, sin_phase]
    along the channel axis → 3 * out_channels output channels. Phase is
    computed per bin as arctan2(im, re) and represented in the unit circle
    so downstream layers see a differentiable, wrap-free signal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 128,
        sample_rate: int = 200,
        w0: float = 6.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        use_phase: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_phase = use_phase

        if freq_min is None:
            freq_min = 2.0 if sample_rate <= 200 else 20.0
        if freq_max is None:
            freq_max = sample_rate / 4.0
        if freq_min <= 0 or freq_max <= freq_min:
            raise ValueError(f"Invalid freq range: freq_min={freq_min}, freq_max={freq_max}")

        self.sample_rate = sample_rate
        self.w0 = w0
        self.freq_min = freq_min
        self.freq_max = freq_max
        scales = (w0 / (2 * math.pi)) / torch.logspace(
            math.log10(freq_min), math.log10(freq_max), steps=out_channels
        )
        # Keep init scales as a buffer so subclasses and logging hooks can
        # read the original frequencies without recomputing them.
        self.register_buffer("init_scales", scales.clone())
        kernel_re, kernel_im = self._build_kernels(scales, w0)
        self.register_buffer("kernel_re", kernel_re)
        self.register_buffer("kernel_im", kernel_im)

    @property
    def total_out_channels(self) -> int:
        """Channels emitted by forward. 3x when use_phase=True."""
        return 3 * self.out_channels if self.use_phase else self.out_channels

    def _time_grid(self) -> torch.Tensor:
        """Kernel-support time grid in seconds. Shared between fixed and
        learnable paths. Separated so subclasses can rebuild kernels on
        every forward pass."""
        t = (
            torch.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size).float()
            / self.sample_rate
        )
        return t

    def _build_kernels(
        self,
        scales: torch.Tensor,
        w0: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build real/imaginary Morlet kernels from scales and w0.

        scales: (out_channels,) tensor. Kept as tensor throughout — do NOT
            call .item() per-element; that would detach autograd on any
            subclass that parameterizes scales. The tensorized form works
            identically for buffers (fixed variant).
        w0: scalar or (out_channels,) tensor. Per-filter w0 enables the
            learnable-w0 variant without changing the fixed-variant call
            site (passes a Python float, broadcasts cleanly).

        t must be in seconds so (t/s) is dimensionless (s is w0/(2π·freq)
        which has units of seconds). The old version used t in samples,
        which caused kernels to underflow to zero at SR ≥ ~400 with any
        non-default freq range — see commit message for details.
        """
        t = self._time_grid().to(scales.device)  # (ks,)
        s = scales.unsqueeze(-1)  # (out, 1)
        # w0 broadcasts: either scalar or (out,) → (out, 1)
        if isinstance(w0, torch.Tensor):
            w0 = w0.to(scales.device).unsqueeze(-1) if w0.ndim == 1 else w0
        norm = (math.pi * s) ** -0.25  # (out, 1)
        gauss = torch.exp(-0.5 * (t / s) ** 2)  # (out, ks)
        kernel_re = norm * gauss * torch.cos(w0 * t / s)  # (out, ks)
        kernel_im = norm * gauss * torch.sin(w0 * t / s)  # (out, ks)
        # Broadcast to (out, in, ks) to match conv1d weight shape.
        kernel_re = kernel_re.unsqueeze(1).expand(-1, self.in_channels, -1)
        kernel_im = kernel_im.unsqueeze(1).expand(-1, self.in_channels, -1)
        return kernel_re, kernel_im

    # FFT-conv threshold: kernel sizes at or above this switch to the FFT
    # path, which is O(N log N) in signal length and beats cuDNN's direct
    # conv for large kernels. Below the threshold, cuDNN wins.
    # Empirically tuned: audio Morlet (ks=4585) is ~100x faster via FFT;
    # seismic Morlet (ks=573) is also faster via FFT; small kernels (ks<256)
    # regress. 512 is a safe cutoff.
    FFT_CONV_THRESHOLD = 512

    def _apply_conv_and_postprocess(
        self,
        x: torch.Tensor,
        kernel_re: torch.Tensor,
        kernel_im: torch.Tensor,
    ) -> torch.Tensor:
        """Shared conv + log-power/phase pipeline. Called from both the
        fixed-variant forward (using buffers) and learnable-variant
        forward (using parameter-derived kernels)."""
        if self.kernel_size >= self.FFT_CONV_THRESHOLD:
            re_out, im_out = self._fft_morlet_conv(x, kernel_re, kernel_im)
        else:
            re_out = F.conv1d(x, kernel_re, padding=self.padding)
            im_out = F.conv1d(x, kernel_im, padding=self.padding)

        if self.use_phase:
            # Compute phase BEFORE squaring re/im (pow_ mutates them).
            mag = torch.sqrt(re_out.pow(2) + im_out.pow(2) + 1e-8)
            cos_phase = re_out / mag
            sin_phase = im_out / mag
            log_power = torch.log1p(re_out.pow(2) + im_out.pow(2))
            return torch.cat([log_power, cos_phase, sin_phase], dim=1)

        power = re_out.pow(2) + im_out.pow(2)
        return torch.log1p(power)

    def _fft_morlet_conv(
        self,
        x: torch.Tensor,
        kernel_re: torch.Tensor,
        kernel_im: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FFT-based Morlet cross-correlation.

        Computes the equivalent of
            re_out = F.conv1d(x, kernel_re, padding=ks//2)
            im_out = F.conv1d(x, kernel_im, padding=ks//2)
        using one complex FFT pair per input channel, giving O(N log N)
        scaling in signal length instead of O(N·K).

        Input shapes:
          x          : (B, in_channels, L)
          kernel_re  : (out_channels, in_channels, ks)
          kernel_im  : same shape as kernel_re

        Output shapes:
          re_out, im_out : each (B, out_channels, L)

        Math: PyTorch conv1d computes cross-correlation
          y[t] = sum_tau x[t+tau] * k[tau]
        which in frequency domain is irfft(rfft(x) * conj(rfft(k))).
        Bundling re + im kernels as complex k = kernel_re + i·kernel_im
        lets one complex multiply produce both outputs.
        """
        B, C_in, L = x.shape
        C_out, _, ks = kernel_re.shape
        assert ks % 2 == 1, f"FFT conv path requires odd kernel_size, got {ks}"

        # FFT length: L + ks - 1 covers full linear cross-correlation,
        # rounded to next power of 2 for a faster transform.
        n_fft = 1
        while n_fft < L + ks - 1:
            n_fft <<= 1

        # rFFT of the real input and both real kernels. Using rFFT (not FFT)
        # saves ~half the cost on real signals. Shape (..., n_fft // 2 + 1).
        X = torch.fft.rfft(x, n=n_fft, dim=-1)
        K_re_f = torch.fft.rfft(kernel_re, n=n_fft, dim=-1)
        K_im_f = torch.fft.rfft(kernel_im, n=n_fft, dim=-1)

        # Cross-correlation in frequency domain: Y = X * conj(K).
        # Summed over C_in via einsum, for both re and im kernels.
        # (B, C_in, F) × (C_out, C_in, F) → (B, C_out, F).
        Y_re = torch.einsum("bif,oif->bof", X, K_re_f.conj())
        Y_im = torch.einsum("bif,oif->bof", X, K_im_f.conj())

        # Inverse rFFT back to real time domain, length n_fft.
        re_full = torch.fft.irfft(Y_re, n=n_fft, dim=-1)
        im_full = torch.fft.irfft(Y_im, n=n_fft, dim=-1)

        # Align with F.conv1d(padding=ks//2). For odd ks, direct conv
        # centers the kernel at sample 0 (uses samples -ks//2 .. +ks//2,
        # treating out-of-bounds as zero). FFT cross-correlation via
        # conj-multiply gives output[0] = sum_tau x[tau]·k[tau] (kernel
        # anchored at start, not centered). Rolling the output by +ks//2
        # puts the centered-kernel value at index 0, matching conv1d.
        # Zero-padding ensures no wrap-around aliasing into valid output.
        shift = ks // 2
        re_out = torch.roll(re_full, shifts=shift, dims=-1)[..., :L]
        im_out = torch.roll(im_full, shifts=shift, dims=-1)[..., :L]
        return re_out, im_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self._apply_conv_and_postprocess(x, self.kernel_re, self.kernel_im)


class LearnableMorletFilterbank(MorletFilterbank):
    """Morlet filterbank with learnable scales (and optionally per-filter w0).

    Parameterization:
      - log_scales: nn.Parameter, shape (out_channels,). scales = exp(log_scales)
        at every forward pass. Log-space keeps scales positive without a hard
        clamp and gives gradients a well-behaved magnitude.
      - w0_per_filter: nn.Parameter, shape (out_channels,), only when
        learnable_w0=True. Otherwise w0 is the scalar inherited from parent.

    Initialization exactly matches MorletFilterbank for the same (freq_min,
    freq_max, w0, out_channels, sample_rate). This means epoch-0 output is
    bit-equivalent — the subclass adds learnability without perturbing the
    starting point.

    The parent's kernel_re / kernel_im buffers are removed in __init__ since
    kernels are rebuilt on every forward pass from current parameter values.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 128,
        sample_rate: int = 200,
        w0: float = 6.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        use_phase: bool = False,
        learnable_w0: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            w0=w0,
            freq_min=freq_min,
            freq_max=freq_max,
            use_phase=use_phase,
        )
        # Parent registered kernel_re/kernel_im as buffers; we rebuild them
        # every forward pass from current parameter values, so drop them to
        # keep state_dict clean and avoid accidental reuse.
        del self.kernel_re
        del self.kernel_im

        # log_scales is the actual learnable parameter. scales = exp(log_scales)
        # on every forward — keeps scales positive without clamping.
        self.log_scales = nn.Parameter(torch.log(self.init_scales.clone()))

        self.learnable_w0 = learnable_w0
        if learnable_w0:
            self.w0_per_filter = nn.Parameter(torch.full((out_channels,), float(w0)))
        # else: use self.w0 (scalar float) inherited from parent __init__.

    def current_scales(self) -> torch.Tensor:
        """Current scales (reparameterized from log_scales). Positive by
        construction."""
        return self.log_scales.exp()

    def current_frequencies(self) -> torch.Tensor:
        """Current center frequencies in Hz. freq = w0 / (2π · scale).

        Uses per-filter w0 when learnable_w0=True, otherwise the scalar w0.
        Detached from the graph — this is for logging, not loss computation.
        """
        with torch.no_grad():
            scales = self.current_scales()
            w0_vec = (
                self.w0_per_filter if self.learnable_w0 else torch.full_like(scales, float(self.w0))
            )
            return w0_vec / (2 * math.pi * scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        # Rebuild kernels from current parameter values. This is the hot path —
        # but _build_kernels is vectorized over filters and runs in O(out*ks),
        # which is negligible next to the conv itself.
        scales = self.current_scales()
        w0 = self.w0_per_filter if self.learnable_w0 else self.w0
        kernel_re, kernel_im = self._build_kernels(scales, w0)
        return self._apply_conv_and_postprocess(x, kernel_re, kernel_im)
