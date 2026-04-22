from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScale1DFrontend(nn.Module):
    """Parallel multi-scale Conv1D branches with GroupNorm + GELU, merged by 1x1 conv.
    Runs in fp32 to avoid overflow on large-stride audio inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int] | None = None,
        stride: int = 4,
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [9, 19, 39]
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=ks, stride=stride,
                          padding=ks // 2, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
            ))
        self.proj = nn.Conv1d(
            len(kernel_sizes) * out_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        feats = [branch(x) for branch in self.branches]
        min_len = min(f.shape[-1] for f in feats)
        feats = [f[..., :min_len] for f in feats]
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
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.padding      = kernel_size // 2
        self.use_phase    = use_phase

        if freq_min is None:
            freq_min = 2.0 if sample_rate <= 200 else 20.0
        if freq_max is None:
            freq_max = sample_rate / 4.0
        if freq_min <= 0 or freq_max <= freq_min:
            raise ValueError(
                f"Invalid freq range: freq_min={freq_min}, freq_max={freq_max}"
            )

        self.sample_rate = sample_rate
        scales = (w0 / (2 * math.pi)) / torch.logspace(
            math.log10(freq_min), math.log10(freq_max), steps=out_channels
        )
        kernel_re, kernel_im = self._build_kernels(scales, w0)
        self.register_buffer("kernel_re", kernel_re)
        self.register_buffer("kernel_im", kernel_im)

    @property
    def total_out_channels(self) -> int:
        """Channels emitted by forward. 3x when use_phase=True."""
        return 3 * self.out_channels if self.use_phase else self.out_channels

    def _build_kernels(self, scales: torch.Tensor, w0: float) -> tuple[torch.Tensor, torch.Tensor]:
        # t must be in seconds so (t/s) is dimensionless (s is w0/(2π·freq)
        # which has units of seconds). The old version used t in samples,
        # which caused kernels to underflow to zero at SR ≥ ~400 with any
        # non-default freq range — see commit message for details.
        t = torch.linspace(
            -self.kernel_size // 2, self.kernel_size // 2, self.kernel_size
        ).float() / self.sample_rate
        kernels_re, kernels_im = [], []
        for s in scales:
            norm = (math.pi * s.item()) ** -0.25
            gauss = torch.exp(-0.5 * (t / s) ** 2)
            kernels_re.append((norm * gauss * torch.cos(w0 * t / s)).unsqueeze(0))
            kernels_im.append((norm * gauss * torch.sin(w0 * t / s)).unsqueeze(0))
        re = torch.stack(kernels_re, dim=0).expand(-1, self.in_channels, -1)  # (out, in, ks)
        im = torch.stack(kernels_im, dim=0).expand(-1, self.in_channels, -1)
        return re, im

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        re_out = F.conv1d(x, self.kernel_re, padding=self.padding)
        im_out = F.conv1d(x, self.kernel_im, padding=self.padding)

        if self.use_phase:
            # Compute phase BEFORE squaring re/im (pow_ mutates them).
            mag = torch.sqrt(re_out.pow(2) + im_out.pow(2) + 1e-8)
            cos_phase = re_out / mag
            sin_phase = im_out / mag
            log_power = torch.log1p(re_out.pow(2) + im_out.pow(2))
            return torch.cat([log_power, cos_phase, sin_phase], dim=1)

        # Log-power only — in-place ops are safe here.
        power = re_out.pow_(2).add_(im_out.pow_(2))
        return torch.log1p(power)
