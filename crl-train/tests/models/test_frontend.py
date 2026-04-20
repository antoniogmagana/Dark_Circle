import pytest
import torch
import torch.nn as nn
from crl_vehicle.models.frontend import MultiScale1DFrontend, MorletFilterbank


class TestMultiScale1DFrontend:

    def test_output_shape_audio(self):
        fe = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=64),
            nn.AvgPool1d(kernel_size=16, stride=16),
        )
        with torch.no_grad():
            out = fe(torch.zeros(2, 1, 16000))
        assert out.shape == (2, 64, 250)

    def test_output_shape_seismic(self):
        fe = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=64),
            nn.AvgPool1d(kernel_size=16, stride=16),
        )
        with torch.no_grad():
            out = fe(torch.zeros(2, 1, 200))
        assert out.shape == (2, 64, 3)

    def test_output_channels_configurable(self):
        fe = MultiScale1DFrontend(in_channels=1, out_channels=32)
        with torch.no_grad():
            out = fe(torch.zeros(2, 1, 1000))
        assert out.shape[1] == 32

    def test_no_nans(self):
        fe = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=64),
            nn.AvgPool1d(kernel_size=16, stride=16),
        )
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 16000) * 0.01)
        assert out.isfinite().all()

    def test_groupnorm_batch_size_1(self):
        fe = MultiScale1DFrontend(in_channels=1, out_channels=64)
        with torch.no_grad():
            out = fe(torch.randn(1, 1, 200))
        assert out.isfinite().all()


class TestMorletFilterbank:

    def test_no_learned_parameters(self):
        fe = MorletFilterbank(in_channels=1, out_channels=32, kernel_size=101, sample_rate=200)
        assert sum(p.numel() for p in fe.parameters()) == 0

    def test_kernel_buffer_shape(self):
        fe = MorletFilterbank(in_channels=1, out_channels=32, kernel_size=101, sample_rate=200)
        buffers = dict(fe.named_buffers())
        assert "kernel" in buffers
        # kernel = cat([real, imag]) → (2*out_channels, in_channels, kernel_size)
        assert buffers["kernel"].shape == (64, 1, 101)

    def test_output_finite_audio(self):
        fe = MorletFilterbank(in_channels=1, out_channels=64, kernel_size=257, sample_rate=16000)
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 16000) * 0.01)
        assert out.isfinite().all()

    def test_output_finite_seismic(self):
        fe = MorletFilterbank(in_channels=1, out_channels=64, kernel_size=199, sample_rate=200)
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 200) * 0.01)
        assert out.isfinite().all()


class TestEarlyFusionShapeReconciliation:

    def test_adaptive_pool_equalizes_tokens(self):
        T_fused, d_model = 32, 64
        fe_audio = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=d_model),
            nn.AvgPool1d(16, 16), nn.AdaptiveAvgPool1d(T_fused),
        )
        fe_seismic = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=d_model),
            nn.AvgPool1d(16, 16), nn.AdaptiveAvgPool1d(T_fused),
        )
        with torch.no_grad():
            a = fe_audio(torch.zeros(4, 1, 16000))
            s = fe_seismic(torch.zeros(4, 1, 200))
            fused = torch.cat([a, s], dim=2)
        assert a.shape     == (4, d_model, T_fused)
        assert s.shape     == (4, d_model, T_fused)
        assert fused.shape == (4, d_model, 2 * T_fused)
        assert fused.isfinite().all()
