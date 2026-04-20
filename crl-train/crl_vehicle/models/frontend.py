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
    Registers combined real+imag kernel as a buffer; forward returns log-power envelope."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 128,
        sample_rate: int = 200,
        w0: float = 6.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        freq_min = 2.0 if sample_rate <= 200 else 20.0
        freq_max = sample_rate / 4.0
        scales = (w0 / (2 * math.pi)) / torch.logspace(
            math.log10(freq_min), math.log10(freq_max), steps=out_channels
        )
        kernel_re, kernel_im = self._build_kernels(scales, w0)
        self.register_buffer("kernel_re", kernel_re)
        self.register_buffer("kernel_im", kernel_im)

    def _build_kernels(self, scales: torch.Tensor, w0: float) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(
            -self.kernel_size // 2, self.kernel_size // 2, self.kernel_size
        ).float()
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
        # Apply real and imaginary kernels separately to avoid materializing
        # the full (B, 2*out, T) intermediate — peak activation is halved.
        re_out = F.conv1d(x, self.kernel_re, padding=self.padding)
        im_out = F.conv1d(x, self.kernel_im, padding=self.padding)
        power = re_out.pow_(2).add_(im_out.pow_(2))
        return torch.log1p(power)
