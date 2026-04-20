import pytest
import torch
from crl_vehicle.models.encoder_decoder import TemporalEncoder, FeatureDecoder


class TestTemporalEncoder:

    @pytest.fixture
    def enc(self):
        return TemporalEncoder(in_channels=64, d_z=24, d_model=64, n_heads=4, n_layers=2)

    def test_output_shapes(self, enc):
        enc.train()
        z, mu, logvar = enc(torch.randn(4, 64, 64))
        assert z.shape == mu.shape == logvar.shape == (4, 24)

    def test_z_equals_mu_in_eval(self, enc):
        enc.eval()
        with torch.no_grad():
            z, mu, _ = enc(torch.randn(4, 64, 64))
        assert torch.allclose(z, mu)

    def test_mu_clamped_to_10(self, enc):
        enc.eval()
        with torch.no_grad():
            _, mu, _ = enc(torch.ones(2, 64, 64) * 100.0)
        assert (mu.abs() <= 10.0 + 1e-5).all()

    def test_logvar_clamped(self, enc):
        enc.eval()
        with torch.no_grad():
            _, _, logvar = enc(torch.ones(2, 64, 64) * 100.0)
        assert (logvar >= -4.0 - 1e-5).all()
        assert (logvar <=  4.0 + 1e-5).all()

    def test_finite(self, enc):
        enc.eval()
        with torch.no_grad():
            z, mu, logvar = enc(torch.randn(4, 64, 64))
        assert z.isfinite().all() and mu.isfinite().all() and logvar.isfinite().all()

    def test_various_seq_lengths(self):
        for T in [3, 32, 64, 250]:
            enc = TemporalEncoder(in_channels=64, d_z=24, d_model=32, n_heads=4, n_layers=1)
            enc.eval()
            with torch.no_grad():
                z, _, _ = enc(torch.randn(2, 64, T))
            assert z.shape == (2, 24), f"Failed for T={T}"


class TestFeatureDecoder:

    def test_output_shape(self):
        dec = FeatureDecoder(out_channels=64, seq_len=64, d_z=24, d_model=64)
        with torch.no_grad():
            out = dec(torch.randn(4, 24))
        assert out.shape == (4, 64, 64)

    def test_finite(self):
        dec = FeatureDecoder(out_channels=64, seq_len=64, d_z=24, d_model=64)
        with torch.no_grad():
            out = dec(torch.randn(4, 24))
        assert out.isfinite().all()

    def test_encode_decode_roundtrip_shape(self):
        T = 64
        enc = TemporalEncoder(in_channels=64, d_z=24, d_model=64, n_heads=4, n_layers=1)
        dec = FeatureDecoder(out_channels=64, seq_len=T, d_z=24, d_model=64)
        enc.eval()
        x = torch.randn(2, 64, T)
        with torch.no_grad():
            z, _, _ = enc(x)
            x_hat = dec(z)
        assert x_hat.shape == x.shape
