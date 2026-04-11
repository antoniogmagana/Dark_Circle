"""
SpectrogramFrontend

Fixed spectrogram computation replacing LearnableFilterbank.

Audio  : raw waveform (B, C, W) → log-mel spectrogram  (B, n_mels*C,    T')
Seismic: raw waveform (B, C, W) → log-STFT magnitude   (B, (n_fft//2+1)*C, T')

Mel scale is appropriate for audio (16 kHz) where vehicle acoustic energy
spans 50 Hz – 8 kHz and perceptual weighting aids discrimination.

Linear STFT is used for seismic (200 Hz) because mel compression non-linearly
distorts the low-frequency bands (2–90 Hz) that carry ground-borne vehicle
vibration — the diagnostic content is concentrated there and must remain
uniformly resolved.

Shape contract (same as former LearnableFilterbank):
    Input : (B, C, W)      — raw waveform, channels-first
    Output: (B, F*C, T')   — log1p-compressed spectrogram, freq-first

where F = n_mels (audio) or n_fft//2+1 (seismic), T' = W//hop_length + 1.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as AT

from crl_vehicle.config import ModalityConfig


class SpectrogramFrontend(nn.Module):
    """
    Fixed (non-learnable) spectrogram frontend for one sensor modality.

    Args:
        mod_cfg : ModalityConfig — provides n_fft, hop_length, n_mels,
                  f_min, f_max, sample_rate
    """

    def __init__(self, mod_cfg: ModalityConfig):
        super().__init__()
        self._mod_cfg = mod_cfg

        if mod_cfg.n_mels > 0:
            self.transform = AT.MelSpectrogram(
                sample_rate=mod_cfg.sample_rate,
                n_fft=mod_cfg.n_fft,
                hop_length=mod_cfg.hop_length,
                n_mels=mod_cfg.n_mels,
                f_min=mod_cfg.f_min,
                f_max=mod_cfg.f_max if mod_cfg.f_max > 0 else None,
                center=True,
                power=2.0,
            )
        else:
            self.transform = AT.Spectrogram(
                n_fft=mod_cfg.n_fft,
                hop_length=mod_cfg.hop_length,
                center=True,
                power=2.0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, W)
        Returns: (B, F*C, T') — log1p-compressed spectrogram
        """
        B, C, W = x.shape
        # Flatten channels into batch, transform, then restore
        x_flat = x.reshape(B * C, W)             # (B*C, W)
        spec = self.transform(x_flat)             # (B*C, F, T')
        F, T = spec.shape[-2], spec.shape[-1]
        spec = spec.reshape(B, C * F, T)          # (B, F*C, T')
        return torch.log1p(spec)

    def center_frequencies(self) -> torch.Tensor:
        """
        Return a 1-D CPU float tensor of frequency bin centres in Hz.

        Audio  (n_mels > 0): mel-scale centre frequencies derived from the
                              mel filterbank used at construction time.
        Seismic (n_mels == 0): linear STFT bins from 0 Hz to Nyquist.
        """
        import torchaudio.functional as AF

        cfg = self._mod_cfg
        if cfg.n_mels > 0:
            # melscale_fbanks returns (n_freqs, n_mels); each column is one filter.
            # The centre frequency of filter k is the linear-Hz bin with the
            # highest weight in that column.
            n_freqs = cfg.n_fft // 2 + 1
            f_max = cfg.f_max if cfg.f_max > 0 else cfg.sample_rate / 2.0
            fb = AF.melscale_fbanks(
                n_freqs=n_freqs,
                f_min=cfg.f_min,
                f_max=f_max,
                n_mels=cfg.n_mels,
                sample_rate=cfg.sample_rate,
            )  # (n_freqs, n_mels)
            linear_hz = torch.linspace(0.0, cfg.sample_rate / 2.0, n_freqs)
            peak_bins = fb.argmax(dim=0)          # (n_mels,)
            return linear_hz[peak_bins].float()   # (n_mels,) CPU tensor
        else:
            n_bins = cfg.n_fft // 2 + 1
            return torch.linspace(0.0, cfg.sample_rate / 2.0, n_bins).float()
