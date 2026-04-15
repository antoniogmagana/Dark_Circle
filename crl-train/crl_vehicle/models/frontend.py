

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
        self.branches = nn.ModuleList([
            nn.Sequential(
                # padding = x // 2 is critical! It ensures that regardless of the kernel size,
                # every branch outputs the exact same temporal sequence length, allowing
                # them to be concatenated later.
                nn.Conv1d(in_channels=self.in_channels, out_channels=branch_channels, kernel_size=x, stride=strides, padding=x // 2),
                nn.BatchNorm1d(branch_channels),
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
        # Pass the input through each branch independently.
        # Then, concatenate the results along the channel dimension (dim=1).
        # Because of our padding strategy, the temporal dimension (dim=2) matches perfectly.
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        
        # Mix the multi-scale features together and project to exactly out_channels.
        return self.project(x)
    

class LearnableMorlet1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=128, sample_rate=200):
        """
        Learnable 1D Morlet Wavelet frontend.

        This layer learns to apply a bank of `out_channels` complex Morlet wavelets
        to the input signal. The 'scale' of each wavelet (which is inversely
        proportional to its frequency) is a learnable parameter.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of wavelet filters to learn.
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
        
        # Create a fixed time vector `t` for the wavelet kernel, centered at 0.
        # This represents the time axis of the kernel in seconds.
        # It's stored as a buffer because it's fixed state, not a learnable parameter.
        t = torch.arange(-(kernel_size // 2), kernel_size // 2, dtype=torch.float32) / sample_rate
        self.register_buffer('t', t)
        
        # These are the learnable scale parameters `s` for each of the `out_channels` wavelets. We
        # initialize them to cover a sensible frequency range for the given sample rate.
        # The scale `s` (in seconds) is related to frequency `f` (in Hz) by: s = w0 / (2*pi*f)

        # Define frequency ranges of interest based on sensor type
        if self.sample_rate > 1000:  # Heuristic for audio
            f_min, f_max = 20.0, 8000.0  # 20 Hz to 8 kHz (Nyquist for 16k SR)
        else:  # Heuristic for seismic
            f_min, f_max = 2.0, 100.0   # 2 Hz to 100 Hz (Nyquist for 200 SR)

        # Convert frequencies to scales
        s_max = self.w0 / (2 * math.pi * f_min)  # Low frequency -> high scale
        s_min = self.w0 / (2 * math.pi * f_max)  # High frequency -> low scale

        # We initialize the learnable parameters in the inverse-softplus space
        # so that after `softplus`, they fall into our desired [s_min, s_max] range.
        # inv_softplus(y) = log(exp(y) - 1), which is `torch.log(torch.expm1(y))`.
        # This is numerically stable for small y.
        init_min = torch.log(torch.expm1(torch.tensor(s_min)))
        init_max = torch.log(torch.expm1(torch.tensor(s_max)))

        # Initialize from high freq (low scale) to low freq (high scale)
        s_init = torch.linspace(init_min, init_max, self.out_channels)
        self.scales = nn.Parameter(s_init)

    def _build_wavelet_kernels(self):
        """
        Dynamically builds the real and imaginary wavelet kernels from the current `scales`.
        This function is called in every forward pass, so the kernels update as the
        `scales` parameter is optimized.
        """
        # Ensure scales are always positive using softplus. A scale of 0 or less is invalid.
        scales = torch.nn.functional.softplus(self.scales)
        # Reshape for broadcasting: (out_channels, 1)
        scales = scales.view(-1, 1)

        # Scaled time `t/s` for each wavelet.
        # `t` is (kernel_size,), `scales` is (out_channels, 1).
        # Broadcasting results in a (out_channels, kernel_size) tensor.
        t_scaled = self.t.view(1, -1) / scales

        # Gaussian envelope: exp(-0.5 * (t/s)^2)
        # The `1 / sqrt(s)` term normalizes the energy of the wavelet across scales.
        envelope = torch.exp(-0.5 * (t_scaled ** 2)) / torch.sqrt(scales)

        # Real (cosine) and Imaginary (sine) parts of the wavelet.
        osc_re = torch.cos(self.w0 * t_scaled)
        osc_im = torch.sin(self.w0 * t_scaled)

        # Multiply the oscillations by the envelope to get the final Morlet wavelets.
        kernel_re = envelope * osc_re
        kernel_im = envelope * osc_im

        # Reshape for PyTorch's conv1d, which expects (out_channels, in_channels, kernel_size).
        # We apply the same filter to all input channels (depthwise-style).
        kernel_re = kernel_re.unsqueeze(1).expand(-1, self.in_channels, -1)
        kernel_im = kernel_im.unsqueeze(1).expand(-1, self.in_channels, -1)

        return kernel_re, kernel_im

    def forward(self, x):
        """
        Apply the learnable wavelet transform to the input signal.

        Args:
            x: Input waveform tensor of shape (B, C, W).

        Returns:
            The power envelope of the wavelet-transformed signal, shape (B, C_out, W').
        """
        # Build the real and imaginary kernels on the fly using current `scales`.
        kernel_re, kernel_im = self._build_wavelet_kernels()

        # Apply the convolutions. We use functional conv1d to use our dynamic weights.
        # Padding is set to 'same' to maintain the sequence length.
        conv_re = nn.functional.conv1d(x, kernel_re, padding='same')
        conv_im = nn.functional.conv1d(x, kernel_im, padding='same')

        # Compute the power (magnitude) of the complex-valued result.
        # This is sqrt(real^2 + imag^2), which gives the energy at each frequency band
        # regardless of the signal's phase.
        power = torch.sqrt(conv_re.pow(2) + conv_im.pow(2) + 1e-8)
        return power
