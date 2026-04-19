

import torch
import torch.nn as nn
import math

class MultiScale1DFrontend(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], strides=4):
        """
        Multi-Scale 1D Convolutional Frontend (Early Fusion).

        Applies parallel 1D convolutions with drastically different kernel sizes
        to the raw waveform. This allows the network to natively capture both
        sharp, high-frequency transients (short kernels) and slow, low-frequency
        rumbles (long kernels) simultaneously.

        Args:
            in_channels: Number of input sensor channels.
            out_channels: Total number of features to output.
            kernel_sizes: List of kernel sizes for the parallel branches.
            strides: The stride applied to all branches for temporal downsampling.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Divide the total output channel budget equally among the parallel branches.
        branch_channels = out_channels // len(kernel_sizes)

        # 1x1 Convolution used to "mix" the concatenated features from all branches.
        # It also projects the channel count precisely to `out_channels` to correct
        # any rounding errors if `out_channels` isn't perfectly divisible by the number of branches.
        self.project = nn.Conv1d(in_channels=branch_channels * len(kernel_sizes), out_channels=out_channels, kernel_size=1)

        # Create the parallel convolutional branches.
        # GroupNorm instead of BatchNorm1d: no running stats means a single inf batch
        # cannot poison subsequent forward passes, and it is stable across the mixed-
        # dataset batches produced by stratified partner sampling.
        # num_groups=1 is equivalent to LayerNorm over channels; kept small so it works
        # even at batch_size=1 during smoke tests.
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=branch_channels, kernel_size=x, stride=strides, padding=x // 2),
                nn.GroupNorm(num_groups=max(g for g in range(1, min(8, branch_channels) + 1) if branch_channels % g == 0), num_channels=branch_channels),
                nn.GELU(),
            ) for x in kernel_sizes
        ])

    def forward(self, x):
        """
        Args:
            x: Raw waveform tensor of shape (B, C_in, W)

        Returns:
            Mixed multi-scale features of shape (B, C_out, W // strides)
        """
        # fp32 throughout: audio windows (16 000 samples) produce conv outputs that
        # overflow fp16 (max ~65 504) at large strides, yielding inf → NaN in GroupNorm.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            x = torch.cat([branch(x) for branch in self.branches], dim=1)
            return self.project(x)


class LearnableMorlet1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=128, sample_rate=200):
        """
        Fixed 1D Morlet Wavelet filterbank frontend.

        Applies a bank of `out_channels` complex Morlet wavelets to the input
        signal. The wavelet scales are fixed at init (log-spaced to cover the
        full frequency range of the sensor) and stored as buffers rather than
        learnable parameters. This makes the filterbank a deterministic
        featurizer — the encoder layers downstream learn which frequency bands
        are discriminative.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of wavelet filters.
            kernel_size: The length of the wavelet kernels in samples.
            sample_rate: The sample rate of the input waveform in Hz.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # w0 is the dimensionless center frequency of the mother wavelet.
        # A value of ~6.0 provides a good balance between time and frequency resolution
        # and satisfies the wavelet admissibility condition.
        self.w0 = 6.0

        # Fixed time vector for the wavelet kernel, centered at 0 (seconds).
        t_vec = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, dtype=torch.float32) / sample_rate
        self.register_buffer('t', t_vec)

        # Define frequency ranges of interest based on sensor type.
        if self.sample_rate > 1000:  # Heuristic for audio
            f_min, f_max = 20.0, 8000.0  # 20 Hz to 8 kHz (Nyquist for 16k SR)
        else:  # Heuristic for seismic
            f_min, f_max = 2.0, 100.0   # 2 Hz to 100 Hz (Nyquist for 200 SR)

        # Convert log-spaced frequencies to scales: s = w0 / (2*pi*f)
        s_max = self.w0 / (2 * math.pi * f_min)
        s_min = self.w0 / (2 * math.pi * f_max)
        scales = torch.exp(torch.linspace(math.log(s_min), math.log(s_max), out_channels))

        # Build kernels once at init and store as buffers (no gradient, no rebuild per step).
        kernel_re, kernel_im = self._build_wavelet_kernels(scales)
        self.register_buffer('kernel_re', kernel_re)
        self.register_buffer('kernel_im', kernel_im)

    def _build_wavelet_kernels(self, scales):
        """
        Build real and imaginary Morlet wavelet kernels from a scale tensor.

        Args:
            scales: 1-D float32 tensor of shape (out_channels,) with positive scale values.

        Returns:
            kernel_re, kernel_im: tensors of shape (out_channels, in_channels, kernel_size)
        """
        scales = scales.float().view(-1, 1)

        # t/s for each wavelet. Broadcasting: (out_channels, kernel_size).
        t_scaled = self.t.float().view(1, -1) / scales

        # Clamp to prevent trig NaN from any extreme scale values.
        # At |t_scaled| = 25, exp(-0.5 * 625) < 1e-135 — effectively zero.
        t_scaled = t_scaled.clamp(-25.0, 25.0)

        # Gaussian envelope with energy normalization across scales.
        envelope = torch.exp(-0.5 * (t_scaled ** 2)) / torch.sqrt(scales)

        osc_re = torch.cos(self.w0 * t_scaled)
        osc_im = torch.sin(self.w0 * t_scaled)

        # Shape: (out_channels, in_channels, kernel_size)
        kernel_re = (envelope * osc_re).unsqueeze(1).expand(-1, self.in_channels, -1).contiguous()
        kernel_im = (envelope * osc_im).unsqueeze(1).expand(-1, self.in_channels, -1).contiguous()

        return kernel_re, kernel_im

    def forward(self, x):
        """
        Apply the fixed wavelet filterbank to the input signal.

        Args:
            x: Input waveform tensor of shape (B, C, W).

        Returns:
            Power envelope of shape (B, out_channels, W).
        """
        # Run in fp32 regardless of AMP context. functional.conv1d with
        # dynamically-supplied weight tensors is cast by input dtype, not the
        # AMP allowlist. Audio scales as small as ~1e-4 s can produce t_scaled
        # values up to ~526 — fine in fp32 but overflowing fp16 (max ~65504).
        with torch.amp.autocast("cuda", enabled=False):
            conv_re = nn.functional.conv1d(x.float(), self.kernel_re, padding='same')
            conv_im = nn.functional.conv1d(x.float(), self.kernel_im, padding='same')
            power = torch.sqrt(conv_re.pow(2) + conv_im.pow(2) + 1e-8)
        return torch.log1p(power)
