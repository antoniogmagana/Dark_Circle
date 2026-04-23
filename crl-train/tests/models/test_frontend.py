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
        assert "kernel_re" in buffers and "kernel_im" in buffers
        assert buffers["kernel_re"].shape == (32, 1, 101)
        assert buffers["kernel_im"].shape == (32, 1, 101)

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


class TestMorletExplicitFreqRange:
    """freq_min/freq_max overrides should let audio/seismic banks cover
    different discriminative bands. The whole point of Checkpoint 2's
    morlet_per_sensor is that the shared-bank heuristic was wrong."""

    def test_explicit_freq_range_applied(self):
        fe1 = MorletFilterbank(
            in_channels=1, out_channels=16, kernel_size=101, sample_rate=16000,
            freq_min=100.0, freq_max=1000.0,
        )
        fe2 = MorletFilterbank(
            in_channels=1, out_channels=16, kernel_size=101, sample_rate=16000,
            freq_min=2000.0, freq_max=8000.0,
        )
        # Different freq ranges ⇒ different kernels.
        assert not torch.equal(fe1.kernel_re, fe2.kernel_re)

    def test_invalid_freq_range_raises(self):
        with pytest.raises(ValueError, match="Invalid freq range"):
            MorletFilterbank(1, 8, kernel_size=51, sample_rate=200,
                             freq_min=100.0, freq_max=50.0)
        with pytest.raises(ValueError, match="Invalid freq range"):
            MorletFilterbank(1, 8, kernel_size=51, sample_rate=200,
                             freq_min=0.0, freq_max=50.0)

    def test_sr_heuristic_still_works_when_none(self):
        """Backward compat: no explicit freq_min/freq_max → SR heuristic."""
        fe = MorletFilterbank(1, 16, kernel_size=101, sample_rate=200)
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 200) * 0.01)
        assert out.shape == (2, 16, 200)
        assert out.isfinite().all()

    @pytest.mark.parametrize("sample_rate,freq_range,kernel_size", [
        (200,   (None,   None),    101),   # default (seismic)
        (200,   (5.0,    40.0),    101),   # explicit seismic-like
        (16000, (None,   None),    257),   # default (audio)
        (16000, (20.0,   8000.0),  257),   # explicit audio-like
        (16000, (100.0,  1000.0),  101),   # narrow mid-band
    ])
    def test_kernels_not_underflowed(self, sample_rate, freq_range, kernel_size):
        """Regression: kernels must carry non-negligible energy.

        Catches a class of bug where the Morlet time grid is in samples
        but the scale is in seconds, causing exp(-0.5*(t/s)^2) to underflow
        to zero everywhere except t=0. Old behavior silently produced
        near-delta kernels at SR ≥ ~400 with any non-default freq range."""
        freq_min, freq_max = freq_range
        fe = MorletFilterbank(
            in_channels=1, out_channels=16,
            kernel_size=kernel_size, sample_rate=sample_rate,
            freq_min=freq_min, freq_max=freq_max,
        )
        # Every bin's kernel magnitude must exceed a small floor — not a
        # degenerate near-delta impulse.
        per_bin_energy = fe.kernel_re.pow(2).sum(dim=(-1, -2))
        assert (per_bin_energy > 1e-3).all(), (
            f"Kernels underflowed for sample_rate={sample_rate}, "
            f"freq_range={freq_range}, ks={kernel_size}. "
            f"per_bin_energy={per_bin_energy}"
        )


class TestMorletPhaseChannels:
    def test_phase_off_default(self):
        fe = MorletFilterbank(1, 32, kernel_size=101, sample_rate=200)
        assert fe.use_phase is False
        assert fe.total_out_channels == 32

    def test_phase_on_triples_channels(self):
        fe = MorletFilterbank(1, 32, kernel_size=101, sample_rate=200, use_phase=True)
        assert fe.total_out_channels == 96
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 200) * 0.01)
        assert out.shape == (2, 96, 200)
        assert out.isfinite().all()

    def test_phase_channels_are_unit_circle_where_magnitude_is_nonzero(self):
        """cos²+sin² ≈ 1 per bin where magnitude is meaningful.

        Bins with near-zero filter response produce small re/im both, and
        the +1e-8 epsilon in the normalization suppresses phase to ~0 for
        those — which is correct (phase of no-signal is undefined). Only
        check bins where log_power is above a small threshold."""
        fe = MorletFilterbank(1, 8, kernel_size=51, sample_rate=200, use_phase=True)
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 200))
        log_power = out[:, :8, :]
        cos = out[:, 8:16, :]
        sin = out[:, 16:24, :]
        mask = log_power > 1e-4   # bins with actual signal
        if mask.any():
            norms = cos.pow(2) + sin.pow(2)
            active = norms[mask]
            assert torch.allclose(active, torch.ones_like(active), atol=1e-3)


class TestMorletPerSensorInModel:
    """frontend_type='morlet_per_sensor' must construct CRLModel correctly
    and accept the expected input shapes per sensor."""

    def test_model_constructs(self):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32, n_layers=1, n_heads=4,
            frontend_type="morlet_per_sensor", d_z=24,
            morlet_kernel_size=101,
        )
        model = CRLModel(cfg)
        assert "audio" in model.frontends
        assert "seismic" in model.frontends
        assert "audio" in model.encoders
        assert "seismic" in model.encoders

    def test_forward_runs(self):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32, n_layers=1, n_heads=4,
            frontend_type="morlet_per_sensor", d_z=24,
            morlet_kernel_size=101,
        )
        model = CRLModel(cfg)
        with torch.no_grad():
            _, z_a, _, _ = model.encode("audio",   torch.randn(2, 1, 16000) * 0.01)
            _, z_s, _, _ = model.encode("seismic", torch.randn(2, 1, 200) * 0.01)
        assert z_a.shape == (2, 24)
        assert z_s.shape == (2, 24)

    def test_missing_params_raises(self):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32, n_layers=1, frontend_type="morlet_per_sensor",
            morlet_kernel_size=51,
        )
        # Remove the seismic entry → construction must fail clearly.
        cfg.morlet_per_sensor_params = {
            "audio": {"freq_min": 20.0, "freq_max": 8000.0,
                      "out_channels_frac": 1.0, "w0": 6.0},
        }
        with pytest.raises(ValueError, match="morlet_per_sensor requires params"):
            CRLModel(cfg)

    def test_out_channels_frac_scales(self):
        """out_channels_frac=0.5 should halve the per-sensor channel count."""
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend import MorletFilterbank
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32, n_layers=1, frontend_type="morlet_per_sensor",
            morlet_kernel_size=51,
        )
        cfg.morlet_per_sensor_params = {
            "audio":   {"freq_min": 20.0, "freq_max": 8000.0,
                        "out_channels_frac": 0.5, "w0": 6.0},
            "seismic": {"freq_min": 2.0,  "freq_max": 40.0,
                        "out_channels_frac": 1.0, "w0": 6.0},
        }
        model = CRLModel(cfg)
        # Inspect the Morlet bank inside the audio frontend sequential.
        audio_bank = model.frontends["audio"][0]
        assert isinstance(audio_bank, MorletFilterbank)
        assert audio_bank.out_channels == 16   # 32 * 0.5
        seismic_bank = model.frontends["seismic"][0]
        assert seismic_bank.out_channels == 32  # 32 * 1.0


class TestMorletPerSensorDerivation:
    """Per-sensor coupled derivation of pool_stride and kernel_size."""

    @pytest.mark.parametrize(
        "sensor,SR,W,freq_min,expected_stride,expected_ks_range",
        [
            ("audio",   16000, 16000, 20.0, 500, (4583, 4587)),
            ("seismic",   200,   200,  2.0,   6, (571,  575)),
        ],
    )
    def test_stride_and_kernel_derivation(
        self, sensor, SR, W, freq_min, expected_stride, expected_ks_range
    ):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=16, n_layers=1, n_heads=4,
            frontend_type="morlet_per_sensor", d_z=24,
        )
        # Single-sensor config: only the sensor under test in the params dict.
        cfg.morlet_per_sensor_params = {
            sensor: {
                "freq_min": freq_min, "freq_max": SR / 4,
                "out_channels_frac": 1.0, "w0": 6.0,
                "target_tokens": 32, "receptive_cycles": 3.0,
            },
        }
        model = CRLModel(cfg, sensors=[sensor])

        derived = model._morlet_derived_params[sensor]
        assert derived["pool_stride"]   == expected_stride
        lo, hi = expected_ks_range
        assert lo <= derived["kernel_size"] <= hi
        assert derived["kernel_size"] % 2 == 1  # odd
        assert derived["target_tokens"]    == 32
        assert derived["receptive_cycles"] == 3.0

        # Audit-trail sanity: post_pool_tokens and post_pool_rate present.
        assert "post_pool_tokens" in derived
        assert "post_pool_rate"   in derived

    def test_derived_params_absent_for_non_morlet_per_sensor(self):
        """Only morlet_per_sensor populates the derivation dict."""
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=16, n_layers=1, n_heads=4,
            frontend_type="multiscale", fused_seq_len=16, d_z=24,
        )
        model = CRLModel(cfg)
        assert model._morlet_derived_params == {}


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
