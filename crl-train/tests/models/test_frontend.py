from typing import ClassVar

import pytest
import torch
import torch.nn as nn
from crl_vehicle.models.frontend import (
    LearnableMorletFilterbank,
    MorletFilterbank,
    MultiScale1DFrontend,
)


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
            in_channels=1,
            out_channels=16,
            kernel_size=101,
            sample_rate=16000,
            freq_min=100.0,
            freq_max=1000.0,
        )
        fe2 = MorletFilterbank(
            in_channels=1,
            out_channels=16,
            kernel_size=101,
            sample_rate=16000,
            freq_min=2000.0,
            freq_max=8000.0,
        )
        # Different freq ranges ⇒ different kernels.
        assert not torch.equal(fe1.kernel_re, fe2.kernel_re)

    def test_invalid_freq_range_raises(self):
        with pytest.raises(ValueError, match="Invalid freq range"):
            MorletFilterbank(1, 8, kernel_size=51, sample_rate=200, freq_min=100.0, freq_max=50.0)
        with pytest.raises(ValueError, match="Invalid freq range"):
            MorletFilterbank(1, 8, kernel_size=51, sample_rate=200, freq_min=0.0, freq_max=50.0)

    def test_sr_heuristic_still_works_when_none(self):
        """Backward compat: no explicit freq_min/freq_max → SR heuristic."""
        fe = MorletFilterbank(1, 16, kernel_size=101, sample_rate=200)
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 200) * 0.01)
        assert out.shape == (2, 16, 200)
        assert out.isfinite().all()

    @pytest.mark.parametrize(
        "sample_rate,freq_range,kernel_size",
        [
            (200, (None, None), 101),  # default (seismic)
            (200, (5.0, 40.0), 101),  # explicit seismic-like
            (16000, (None, None), 257),  # default (audio)
            (16000, (20.0, 8000.0), 257),  # explicit audio-like
            (16000, (100.0, 1000.0), 101),  # narrow mid-band
        ],
    )
    def test_kernels_not_underflowed(self, sample_rate, freq_range, kernel_size):
        """Regression: kernels must carry non-negligible energy.

        Catches a class of bug where the Morlet time grid is in samples
        but the scale is in seconds, causing exp(-0.5*(t/s)^2) to underflow
        to zero everywhere except t=0. Old behavior silently produced
        near-delta kernels at SR ≥ ~400 with any non-default freq range."""
        freq_min, freq_max = freq_range
        fe = MorletFilterbank(
            in_channels=1,
            out_channels=16,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
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
        mask = log_power > 1e-4  # bins with actual signal
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
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="morlet_per_sensor",
            d_z=24,
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
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="morlet_per_sensor",
            d_z=24,
            morlet_kernel_size=101,
        )
        model = CRLModel(cfg)
        W_a = cfg.modality_cfg("audio").window_size
        W_s = cfg.modality_cfg("seismic").window_size
        with torch.no_grad():
            _, z_a, _, _ = model.encode("audio", torch.randn(2, 1, W_a) * 0.01)
            _, z_s, _, _ = model.encode("seismic", torch.randn(2, 1, W_s) * 0.01)
        assert z_a.shape == (2, 24)
        assert z_s.shape == (2, 24)

    def test_missing_params_raises(self):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            frontend_type="morlet_per_sensor",
            morlet_kernel_size=51,
        )
        # Remove the seismic entry → construction must fail clearly.
        cfg.morlet_per_sensor_params = {
            "audio": {
                "freq_min": 20.0,
                "freq_max": 8000.0,
                "out_channels_frac": 1.0,
                "w0": 6.0,
            },
        }
        with pytest.raises(ValueError, match="requires params for"):
            CRLModel(cfg)

    def test_out_channels_frac_scales(self):
        """out_channels_frac=0.5 should halve the per-sensor channel count."""
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend import MorletFilterbank
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            frontend_type="morlet_per_sensor",
            morlet_kernel_size=51,
        )
        cfg.morlet_per_sensor_params = {
            "audio": {
                "freq_min": 20.0,
                "freq_max": 8000.0,
                "out_channels_frac": 0.5,
                "w0": 6.0,
            },
            "seismic": {
                "freq_min": 2.0,
                "freq_max": 40.0,
                "out_channels_frac": 1.0,
                "w0": 6.0,
            },
        }
        model = CRLModel(cfg)
        # Inspect the Morlet bank inside the audio frontend sequential.
        audio_bank = model.frontends["audio"][0]
        assert isinstance(audio_bank, MorletFilterbank)
        assert audio_bank.out_channels == 16  # 32 * 0.5
        seismic_bank = model.frontends["seismic"][0]
        assert seismic_bank.out_channels == 32  # 32 * 1.0


class TestMorletPerSensorDerivation:
    """Per-sensor coupled derivation of pool_stride and kernel_size."""

    @pytest.mark.parametrize(
        "sensor,SR,W,freq_min,expected_stride,expected_ks_range",
        [
            # SR/W are the canonical post-resample rates returned by
            # CRLConfig.modality_cfg(sensor). The model derives stride/kernel
            # from those, not from these params — they're informational here.
            ("audio", 16000, 16000, 20.0, 500, (4583, 4587)),
            ("seismic", 100, 100, 2.0, 3, (285, 289)),
        ],
    )
    def test_stride_and_kernel_derivation(
        self, sensor, SR, W, freq_min, expected_stride, expected_ks_range
    ):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            frontend_type="morlet_per_sensor",
            d_z=24,
        )
        # Single-sensor config: only the sensor under test in the params dict.
        cfg.morlet_per_sensor_params = {
            sensor: {
                "freq_min": freq_min,
                "freq_max": SR / 4,
                "out_channels_frac": 1.0,
                "w0": 6.0,
                "target_tokens": 32,
                "receptive_cycles": 3.0,
            },
        }
        model = CRLModel(cfg, sensors=[sensor])

        derived = model._morlet_derived_params[sensor]
        assert derived["pool_stride"] == expected_stride
        lo, hi = expected_ks_range
        assert lo <= derived["kernel_size"] <= hi
        assert derived["kernel_size"] % 2 == 1  # odd
        assert derived["target_tokens"] == 32
        assert derived["receptive_cycles"] == 3.0

        # Audit-trail sanity: post_pool_tokens and post_pool_rate present.
        assert "post_pool_tokens" in derived
        assert "post_pool_rate" in derived

    def test_derived_params_absent_for_non_morlet_per_sensor(self):
        """Non-morlet frontends leave the derivation dict empty. (Both
        morlet_per_sensor and morlet_fused populate it.)"""
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            frontend_type="multiscale",
            fused_seq_len=16,
            d_z=24,
        )
        model = CRLModel(cfg)
        assert model._morlet_derived_params == {}


class TestEarlyFusionShapeReconciliation:
    def test_adaptive_pool_equalizes_tokens(self):
        T_fused, d_model = 32, 64
        fe_audio = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=d_model),
            nn.AvgPool1d(16, 16),
            nn.AdaptiveAvgPool1d(T_fused),
        )
        fe_seismic = nn.Sequential(
            MultiScale1DFrontend(in_channels=1, out_channels=d_model),
            nn.AvgPool1d(16, 16),
            nn.AdaptiveAvgPool1d(T_fused),
        )
        with torch.no_grad():
            a = fe_audio(torch.zeros(4, 1, 16000))
            s = fe_seismic(torch.zeros(4, 1, 200))
            fused = torch.cat([a, s], dim=2)
        assert a.shape == (4, d_model, T_fused)
        assert s.shape == (4, d_model, T_fused)
        assert fused.shape == (4, d_model, 2 * T_fused)
        assert fused.isfinite().all()


class TestMorletFusedInModel:
    """frontend_type='morlet_fused' constructs a fused-topology CRLModel
    (single shared encoder/decoder) with per-sensor Morlet banks."""

    def _base_cfg(self, d_model=32):
        from crl_vehicle.config import CRLConfig

        return CRLConfig(
            d_model=d_model,
            n_layers=1,
            n_heads=4,
            frontend_type="morlet_fused",
            fused_seq_len=16,
            d_z=24,
        )

    def test_model_constructs_with_fused_topology(self):
        from training.trainer import CRLModel

        model = CRLModel(self._base_cfg())
        # Per-sensor frontends but a single shared encoder/decoder.
        assert "audio" in model.frontends
        assert "seismic" in model.frontends
        assert model.encoder is not None
        assert model.decoder is not None
        assert len(model.encoders) == 0
        assert len(model.decoders) == 0
        # One shared head set under "fused".
        assert "fused" in model.aux_type_heads
        assert "audio" not in model.aux_type_heads
        assert model.is_fused_frontend() is True

    def test_forward_matches_multiscale_shape(self):
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        model = CRLModel(cfg)
        W_a = cfg.modality_cfg("audio").window_size
        W_s = cfg.modality_cfg("seismic").window_size
        with torch.no_grad():
            features, z, _, _ = model.encode_fused(
                torch.randn(2, 1, W_a) * 0.01,
                torch.randn(2, 1, W_s) * 0.01,
            )
        # Shared encoder sees (B, C, 2 * T) after time-concat.
        assert features.ndim == 3
        assert features.shape[0] == 2
        assert features.shape[2] == 2 * cfg.fused_seq_len
        assert z.shape == (2, 24)
        assert features.isfinite().all()
        assert z.isfinite().all()

    def test_mismatched_out_channels_frac_raises(self):
        """Early fusion concats along time — channel counts must match."""
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        cfg.morlet_per_sensor_params = {
            "audio": {
                "freq_min": 20.0,
                "freq_max": 8000.0,
                "out_channels_frac": 1.0,
                "w0": 6.0,
                "target_tokens": 16,
                "receptive_cycles": 3.0,
            },
            "seismic": {
                "freq_min": 2.0,
                "freq_max": 40.0,
                "out_channels_frac": 0.5,
                "w0": 6.0,
                "target_tokens": 16,
                "receptive_cycles": 3.0,
            },
        }
        with pytest.raises(ValueError, match="matching n_out_channels"):
            CRLModel(cfg)

    def test_missing_params_raises(self):
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        cfg.morlet_per_sensor_params = {
            "audio": {
                "freq_min": 20.0,
                "freq_max": 8000.0,
                "out_channels_frac": 1.0,
                "w0": 6.0,
                "target_tokens": 16,
                "receptive_cycles": 3.0,
            },
        }
        with pytest.raises(ValueError, match="requires params for"):
            CRLModel(cfg)

    def test_derived_params_populated(self):
        """morlet_fused records pool_stride, kernel_size, adaptive_pool_T
        per sensor (same audit trail as morlet_per_sensor, plus adaptive_pool_T)."""
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        model = CRLModel(cfg)
        for sensor in ("audio", "seismic"):
            derived = model._morlet_derived_params[sensor]
            assert "pool_stride" in derived
            assert "kernel_size" in derived
            assert derived["kernel_size"] % 2 == 1  # odd
            assert derived["adaptive_pool_T"] == cfg.fused_seq_len


class TestLearnableMorletFilterbank:
    """LearnableMorletFilterbank parameterizes scales (and optionally w0)
    as nn.Parameter. Epoch-0 output must match the fixed MorletFilterbank
    within float32 precision; gradients must flow to the learnable params."""

    COMMON: ClassVar[dict] = {
        "in_channels": 1,
        "out_channels": 16,
        "kernel_size": 101,
        "sample_rate": 200,
        "w0": 6.0,
        "freq_min": 2.0,
        "freq_max": 40.0,
    }

    @pytest.mark.parametrize("use_phase", [False, True])
    def test_init_matches_fixed_filterbank(self, use_phase):
        """At init (before any gradient step), LearnableMorletFilterbank
        should produce output within float32 precision of MorletFilterbank
        with the same params. The log/exp reparameterization introduces
        ~1e-7 relative error in scales, which amplifies to ~1e-4 in the
        phase-normalized output; non-phase is tighter."""
        torch.manual_seed(0)
        x = torch.randn(2, 1, 1000) * 0.01
        fixed = MorletFilterbank(use_phase=use_phase, **self.COMMON)
        learn = LearnableMorletFilterbank(use_phase=use_phase, **self.COMMON)
        with torch.no_grad():
            y_fixed = fixed(x)
            y_learn = learn(x)
        assert y_fixed.shape == y_learn.shape
        assert y_fixed.dtype == y_learn.dtype
        # Tolerance reflects log/exp round-trip error + downstream mag division.
        atol = 1e-4 if use_phase else 1e-5
        assert torch.allclose(
            y_fixed, y_learn, atol=atol
        ), f"max abs diff {(y_fixed - y_learn).abs().max().item():.3e}"

    def test_gradient_flows_to_log_scales(self):
        """log_scales must receive gradient. This is the regression test
        for the .item() autograd footgun — if _build_kernels ever reverts
        to detaching via .item(), gradients silently stop flowing."""
        learn = LearnableMorletFilterbank(use_phase=False, **self.COMMON)
        x = torch.randn(2, 1, 1000) * 0.01
        loss = learn(x).sum()
        loss.backward()
        assert learn.log_scales.grad is not None
        assert learn.log_scales.grad.abs().sum().item() > 0

    def test_gradient_flows_to_w0_when_learnable(self):
        learn = LearnableMorletFilterbank(
            use_phase=False,
            learnable_w0=True,
            **self.COMMON,
        )
        x = torch.randn(2, 1, 1000) * 0.01
        loss = learn(x).sum()
        loss.backward()
        assert learn.w0_per_filter.grad is not None
        assert learn.w0_per_filter.grad.abs().sum().item() > 0

    def test_w0_not_parameter_when_not_learnable(self):
        learn = LearnableMorletFilterbank(
            use_phase=False,
            learnable_w0=False,
            **self.COMMON,
        )
        assert not hasattr(learn, "w0_per_filter")
        assert isinstance(learn.w0, float)

    def test_scales_stay_positive_after_optimizer_step(self):
        """The log parameterization guarantees exp(log_scales) > 0 even
        after arbitrary gradient steps — including large ones that would
        push raw scales negative."""
        learn = LearnableMorletFilterbank(use_phase=False, **self.COMMON)
        opt = torch.optim.SGD([learn.log_scales], lr=1.0)
        x = torch.randn(2, 1, 1000) * 0.01
        for _ in range(5):
            opt.zero_grad()
            (learn(x).sum()).backward()
            opt.step()
        scales = learn.current_scales()
        assert (scales > 0).all(), f"scales went non-positive: {scales}"

    def test_current_frequencies_reflects_learning(self):
        """If log_scales changes, current_frequencies must change accordingly."""
        learn = LearnableMorletFilterbank(use_phase=False, **self.COMMON)
        f_init = learn.current_frequencies().clone()
        with torch.no_grad():
            learn.log_scales.add_(0.1)  # scales grow → freqs shrink
        f_after = learn.current_frequencies()
        # freq = w0 / (2π·scale); scale grows by factor exp(0.1) ≈ 1.105
        # so freq shrinks by that factor.
        ratio = (f_init / f_after).mean().item()
        assert abs(ratio - torch.exp(torch.tensor(0.1)).item()) < 1e-4

    def test_kernel_buffers_deregistered(self):
        """Parent's kernel_re / kernel_im buffers should not exist on the
        learnable subclass (kernels are rebuilt on every forward pass)."""
        learn = LearnableMorletFilterbank(use_phase=False, **self.COMMON)
        assert "kernel_re" not in dict(learn.named_buffers())
        assert "kernel_im" not in dict(learn.named_buffers())
        # But init_scales (the reference) should still be a buffer.
        assert "init_scales" in dict(learn.named_buffers())


class TestMorletLearnableInModel:
    """frontend_type='morlet_learnable' — late fusion with learnable scales."""

    def _base_cfg(self, **overrides):
        from crl_vehicle.config import CRLConfig

        kwargs = {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "frontend_type": "morlet_learnable",
            "d_z": 24,
        }
        kwargs.update(overrides)
        return CRLConfig(**kwargs)

    def test_constructs_with_per_sensor_topology(self):
        from training.trainer import CRLModel

        model = CRLModel(self._base_cfg())
        assert model.encoder is None
        assert model.decoder is None
        assert "audio" in model.encoders
        assert "seismic" in model.encoders
        assert model.is_fused_frontend() is False

    def test_learnable_parameters_nonempty(self):
        from training.trainer import CRLModel

        model = CRLModel(self._base_cfg())
        params = model.learnable_morlet_parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_forward_runs(self):
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        model = CRLModel(cfg)
        W_a = cfg.modality_cfg("audio").window_size
        W_s = cfg.modality_cfg("seismic").window_size
        with torch.no_grad():
            _, z_a, _, _ = model.encode("audio", torch.randn(2, 1, W_a) * 0.01)
            _, z_s, _, _ = model.encode("seismic", torch.randn(2, 1, W_s) * 0.01)
        assert z_a.shape == (2, 24)
        assert z_s.shape == (2, 24)

    def test_w0_param_added_when_learnable(self):
        from crl_vehicle.models.frontend import LearnableMorletFilterbank
        from training.trainer import CRLModel

        model = CRLModel(self._base_cfg(morlet_learnable_w0=True))
        banks = [
            m
            for stack in model.frontends.values()
            for m in stack.modules()
            if isinstance(m, LearnableMorletFilterbank)
        ]
        assert len(banks) == 2
        for bank in banks:
            assert hasattr(bank, "w0_per_filter")
            assert bank.w0_per_filter.requires_grad


class TestMorletLearnableFusedInModel:
    """frontend_type='morlet_learnable_fused' — early fusion with learnable scales."""

    def _base_cfg(self, **overrides):
        from crl_vehicle.config import CRLConfig

        kwargs = {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "frontend_type": "morlet_learnable_fused",
            "fused_seq_len": 16,
            "d_z": 24,
        }
        kwargs.update(overrides)
        return CRLConfig(**kwargs)

    def test_constructs_with_fused_topology(self):
        from training.trainer import CRLModel

        model = CRLModel(self._base_cfg())
        assert model.encoder is not None
        assert model.decoder is not None
        assert len(model.encoders) == 0
        assert model.is_fused_frontend() is True

    def test_learnable_parameters_nonempty(self):
        from training.trainer import CRLModel

        model = CRLModel(self._base_cfg())
        params = model.learnable_morlet_parameters()
        assert len(params) > 0

    def test_forward_matches_fused_shape(self):
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        model = CRLModel(cfg)
        W_a = cfg.modality_cfg("audio").window_size
        W_s = cfg.modality_cfg("seismic").window_size
        with torch.no_grad():
            features, z, _, _ = model.encode_fused(
                torch.randn(2, 1, W_a) * 0.01,
                torch.randn(2, 1, W_s) * 0.01,
            )
        assert features.shape[0] == 2
        assert features.shape[2] == 2 * cfg.fused_seq_len
        assert z.shape == (2, 24)

    def test_mismatched_fracs_raises(self):
        from training.trainer import CRLModel

        cfg = self._base_cfg()
        cfg.morlet_per_sensor_params = {
            "audio": {
                "freq_min": 20.0,
                "freq_max": 8000.0,
                "out_channels_frac": 1.0,
                "w0": 6.0,
                "target_tokens": 16,
                "receptive_cycles": 3.0,
            },
            "seismic": {
                "freq_min": 2.0,
                "freq_max": 40.0,
                "out_channels_frac": 0.5,
                "w0": 6.0,
                "target_tokens": 16,
                "receptive_cycles": 3.0,
            },
        }
        with pytest.raises(ValueError, match="matching n_out_channels"):
            CRLModel(cfg)

    def test_non_learnable_variant_has_empty_param_list(self):
        """Sanity check: model.learnable_morlet_parameters() returns [] for
        morlet_fused (fixed kernels) — prevents accidental LR-group creation."""
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel

        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            frontend_type="morlet_fused",
            fused_seq_len=16,
            d_z=24,
        )
        model = CRLModel(cfg)
        assert model.learnable_morlet_parameters() == []


class TestLearnableMorletOptimizerGroup:
    """Trainer must place learnable Morlet params in a separate optimizer
    group at lr = backbone_lr * morlet_learnable_lr_mult, and must NOT
    include them in backbone_parameters() (which would double-optimize)."""

    def test_separate_group_for_learnable_variant(self, tmp_path):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel, Trainer

        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            frontend_type="morlet_learnable",
            d_z=24,
            lr=1e-3,
            morlet_learnable_lr_mult=0.1,
        )
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        [g["lr"] for g in trainer.optimizer.param_groups]
        names = [g.get("name", "") for g in trainer.optimizer.param_groups]
        assert "learnable_morlet" in names
        learnable_group = next(
            g for g in trainer.optimizer.param_groups if g.get("name") == "learnable_morlet"
        )
        assert learnable_group["lr"] == 1e-4  # 1e-3 * 0.1

    def test_no_double_optimization(self, tmp_path):
        """Learnable morlet params must appear in exactly one optimizer group."""
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel, Trainer

        cfg = CRLConfig(
            d_model=32,
            n_layers=1,
            frontend_type="morlet_learnable",
            d_z=24,
        )
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        learnable_ids = {id(p) for p in model.learnable_morlet_parameters()}
        appearances = {pid: 0 for pid in learnable_ids}
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in learnable_ids:
                    appearances[id(p)] += 1
        assert all(
            c == 1 for c in appearances.values()
        ), f"param counts per optimizer group: {appearances}"

    def test_no_extra_group_for_fixed_variant(self, tmp_path):
        from crl_vehicle.config import CRLConfig
        from training.trainer import CRLModel, Trainer

        cfg = CRLConfig(d_model=32, n_layers=1, frontend_type="morlet_per_sensor", d_z=24)
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        names = [g.get("name", "") for g in trainer.optimizer.param_groups]
        assert "learnable_morlet" not in names


class TestFFTMorletConv:
    """FFT-based convolution path must match direct F.conv1d within float
    precision and preserve gradients for the learnable variant."""

    def _filterbank(
        self,
        ks,
        use_phase=False,
        sample_rate=200,
        freq_min=2.0,
        freq_max=40.0,
        out_channels=8,
    ):
        return MorletFilterbank(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=ks,
            sample_rate=sample_rate,
            w0=6.0,
            freq_min=freq_min,
            freq_max=freq_max,
            use_phase=use_phase,
        )

    def _force_threshold(self, threshold):
        """Context-manager-style helper for toggling the class-level threshold
        between test runs. Returns the previous value."""
        prev = MorletFilterbank.FFT_CONV_THRESHOLD
        MorletFilterbank.FFT_CONV_THRESHOLD = threshold
        return prev

    @pytest.mark.parametrize(
        "ks,use_phase",
        [
            (513, False),
            (513, True),
            (1025, False),
            # Representative audio-Morlet size — the actual production hot path.
            (4585, False),
        ],
    )
    def test_fft_matches_direct_conv(self, ks, use_phase):
        """Force both paths on the same filterbank and compare outputs."""
        torch.manual_seed(42)
        # Use a longer input than kernel so the conv has non-trivial center.
        L = max(2 * ks, 2000)
        # Pick SR + freq range compatible with kernel_size.
        SR = 16000 if ks > 1000 else 200
        fmin = 20.0 if SR == 16000 else 2.0
        fmax = 8000.0 if SR == 16000 else 40.0
        fb = self._filterbank(ks, use_phase=use_phase, sample_rate=SR, freq_min=fmin, freq_max=fmax)
        x = torch.randn(2, 1, L) * 0.01

        prev = self._force_threshold(10**9)  # disable FFT
        y_direct = fb(x)
        self._force_threshold(0)  # force FFT
        y_fft = fb(x)
        self._force_threshold(prev)

        assert y_direct.shape == y_fft.shape
        # Phase path amplifies numerical diff via 1/sqrt(mag+eps), so its
        # tolerance is looser than the non-phase power path.
        atol = 1e-3 if use_phase else 1e-4
        assert torch.allclose(y_direct, y_fft, atol=atol), (
            f"ks={ks}, use_phase={use_phase}: "
            f"max diff {(y_direct - y_fft).abs().max().item():.3e}"
        )

    def test_threshold_dispatches_correctly(self):
        """Below threshold → direct; at or above → FFT. The dispatch is
        silent, so test it by patching F.conv1d and checking call count."""
        from unittest.mock import patch

        import torch.nn.functional as F_mod

        torch.manual_seed(0)
        x = torch.randn(2, 1, 2000) * 0.01

        # Small kernel → direct path (two conv1d calls, re + im).
        fb_small = self._filterbank(ks=101)  # 101 < 512
        with patch.object(F_mod, "conv1d", wraps=F_mod.conv1d) as spy:
            _ = fb_small(x)
        assert spy.call_count == 2, f"small-kernel direct path, got {spy.call_count}"

        # Large kernel → FFT path (zero conv1d calls).
        fb_large = self._filterbank(ks=513)
        with patch.object(F_mod, "conv1d", wraps=F_mod.conv1d) as spy:
            _ = fb_large(x)
        assert spy.call_count == 0, f"large-kernel FFT path, got {spy.call_count}"

    def test_gradient_flows_through_fft_path(self):
        """LearnableMorletFilterbank with ks above FFT threshold must still
        propagate gradients to log_scales. This is the critical correctness
        check — a broken FFT path silently zeros out learning."""
        learn = LearnableMorletFilterbank(
            in_channels=1,
            out_channels=16,
            kernel_size=513,
            sample_rate=200,
            w0=6.0,
            freq_min=2.0,
            freq_max=40.0,
            use_phase=False,
        )
        assert learn.kernel_size >= MorletFilterbank.FFT_CONV_THRESHOLD
        x = torch.randn(2, 1, 2000) * 0.01
        learn(x).sum().backward()
        assert learn.log_scales.grad is not None
        assert learn.log_scales.grad.abs().sum().item() > 0

    def test_learnable_fft_matches_direct_at_init(self):
        """At init, LearnableMorletFilterbank forward (FFT path) should still
        match the fixed MorletFilterbank forward (direct path). The earlier
        init-matches-fixed test uses ks=101 (direct/direct); this one forces
        ks=513 so we're comparing FFT against direct across subclasses."""
        torch.manual_seed(0)
        fixed = MorletFilterbank(
            in_channels=1,
            out_channels=16,
            kernel_size=513,
            sample_rate=200,
            w0=6.0,
            freq_min=2.0,
            freq_max=40.0,
            use_phase=False,
        )
        learn = LearnableMorletFilterbank(
            in_channels=1,
            out_channels=16,
            kernel_size=513,
            sample_rate=200,
            w0=6.0,
            freq_min=2.0,
            freq_max=40.0,
            use_phase=False,
        )
        x = torch.randn(2, 1, 2000) * 0.01
        with torch.no_grad():
            y_fixed = fixed(x)
            y_learn = learn(x)
        # Loose atol because of log/exp round-trip + FFT rounding combined.
        diff = (y_fixed - y_learn).abs().max().item()
        assert diff < 1e-3, f"max diff {diff:.3e}"


# ---------------------------------------------------------------------------
# MultiScale1DFrontend per-branch strides + target_tokens
# ---------------------------------------------------------------------------


class TestMultiScaleGraduatedStrides:
    """New per-branch stride/alignment API, exercising the SR=4000 experiment shape."""

    def test_per_branch_stride_aligns_via_target_tokens(self):
        kernel_sizes = [201, 41, 9, 5]
        strides = [67, 13, 3, 1]
        fe = MultiScale1DFrontend(
            in_channels=1,
            out_channels=64,
            kernel_sizes=kernel_sizes,
            strides=strides,
            target_tokens=32,
        )
        with torch.no_grad():
            out = fe(torch.randn(2, 1, 4000) * 0.01)
        assert out.shape == (2, 64, 32)
        assert out.isfinite().all()

    def test_all_branches_contribute_when_strides_differ(self):
        """Zero out each branch's first conv weight; output must change.

        Validates the min_len truncation is gone — under the old code the
        4000-token high-freq branch would have been truncated to 60 tokens
        before the proj 1x1 conv saw it, masking changes in that branch.
        """
        torch.manual_seed(0)
        kernel_sizes = [201, 41, 9, 5]
        strides = [67, 13, 3, 1]
        fe = MultiScale1DFrontend(
            in_channels=1,
            out_channels=64,
            kernel_sizes=kernel_sizes,
            strides=strides,
            target_tokens=32,
        )
        x = torch.randn(2, 1, 4000) * 0.01
        with torch.no_grad():
            ref = fe(x).clone()
        for i in range(len(kernel_sizes)):
            saved = fe.branches[i][0].weight.data.clone()
            fe.branches[i][0].weight.data.zero_()
            with torch.no_grad():
                perturbed = fe(x)
            diff = (ref - perturbed).abs().max().item()
            assert diff > 1e-6, f"Branch {i} (ks={kernel_sizes[i]}) had no effect"
            fe.branches[i][0].weight.data.copy_(saved)

    def test_target_tokens_none_preserves_legacy_shape(self):
        """Without target_tokens, the legacy shape (no per-branch pool) holds."""
        fe = MultiScale1DFrontend(in_channels=1, out_channels=64)  # defaults
        with torch.no_grad():
            out = fe(torch.zeros(2, 1, 16000))
        # stride=4 default, kernel_sizes=[9,19,39] default: output = (16000-1)//4 + 1 = 4000
        assert out.shape == (2, 64, 4000)

    def test_strides_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="strides length"):
            MultiScale1DFrontend(
                in_channels=1,
                out_channels=64,
                kernel_sizes=[9, 19, 39],
                strides=[4, 4],  # 2 vs 3
            )


# ---------------------------------------------------------------------------
# Frontend factory (build_frontend) — exercised before trainer wiring
# ---------------------------------------------------------------------------


class TestFrontendFactory:
    """Tests build_frontend in isolation. Not yet wired to trainer."""

    def test_multiscale_early_returns_shared_encoder(self):
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend_factory import build_frontend

        cfg = CRLConfig(d_model=16, n_layers=1, n_heads=4)
        sensors = ["audio", "seismic"]
        frontends, encoder, decoder, encoders, decoders, derived = build_frontend(cfg, sensors)
        assert "audio" in frontends and "seismic" in frontends
        assert encoder is not None and decoder is not None
        assert len(encoders) == 0 and len(decoders) == 0
        # Multiscale leaves derived empty (legacy contract).
        assert derived == {}

    def test_morlet_late_returns_per_sensor_encoders(self):
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend_factory import build_frontend

        cfg = CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            frontend_bank="morlet",
            frontend_fusion="late",
            frontend_per_sensor_params={
                "audio": {
                    "freq_min": 20.0,
                    "freq_max": 8000.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
                "seismic": {
                    "freq_min": 2.0,
                    "freq_max": 40.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
            },
        )
        frontends, encoder, decoder, encoders, decoders, derived = build_frontend(
            cfg, ["audio", "seismic"]
        )
        assert encoder is None and decoder is None
        assert "audio" in encoders and "seismic" in encoders
        assert "audio" in derived and "seismic" in derived
        assert "post_pool_tokens" in derived["audio"]
        assert "kernel_size" in derived["audio"]

    def test_morlet_early_returns_shared_encoder(self):
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend_factory import build_frontend

        cfg = CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            frontend_bank="morlet",
            frontend_fusion="early",
            frontend_per_sensor_params={
                "audio": {
                    "freq_min": 20.0,
                    "freq_max": 8000.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
                "seismic": {
                    "freq_min": 2.0,
                    "freq_max": 40.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
            },
        )
        frontends, encoder, decoder, encoders, decoders, derived = build_frontend(
            cfg, ["audio", "seismic"]
        )
        assert encoder is not None and decoder is not None
        assert len(encoders) == 0 and len(decoders) == 0
        assert "adaptive_pool_T" in derived["audio"]

    def test_morlet_learnable_late_has_learnable_params(self):
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend_factory import build_frontend

        cfg = CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            frontend_bank="morlet_learnable",
            frontend_fusion="late",
            frontend_per_sensor_params={
                "audio": {
                    "freq_min": 20.0,
                    "freq_max": 8000.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
                "seismic": {
                    "freq_min": 2.0,
                    "freq_max": 40.0,
                    "w0": 6.0,
                    "target_tokens": 32,
                    "receptive_cycles": 3.0,
                    "out_channels_frac": 1.0,
                },
            },
        )
        frontends, _, _, _, _, derived = build_frontend(cfg, ["audio", "seismic"])
        # Bank at index 0 of each per-sensor Sequential.
        assert isinstance(frontends["audio"][0], LearnableMorletFilterbank)
        assert derived["audio"]["learnable"] is True

    def test_factory_forward_runs(self):
        """Smoke test: factory output runs through the encoder."""
        from crl_vehicle.config import CRLConfig
        from crl_vehicle.models.frontend_factory import build_frontend

        cfg = CRLConfig(d_model=16, n_layers=1, n_heads=4, audio_target_rate=4000)
        # New SR=4000 graduated kernel config:
        cfg = CRLConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            audio_target_rate=4000,
            frontend_per_sensor_params={
                "audio": {
                    "target_tokens": 32,
                    "kernel_sizes": [201, 41, 9, 5],
                    "strides": [67, 13, 3, 1],
                    "out_channels_frac": 1.0,
                },
                "seismic": {
                    "target_tokens": 32,
                    "kernel_sizes": [9, 19, 39],
                    "out_channels_frac": 1.0,
                },
            },
        )
        frontends, encoder, _, _, _, _ = build_frontend(cfg, ["audio", "seismic"])
        x_audio = torch.randn(2, 1, 4000) * 0.01
        x_seismic = torch.randn(2, 1, 100) * 0.01
        with torch.no_grad():
            f_audio = frontends["audio"](x_audio)
            f_seismic = frontends["seismic"](x_seismic)
        assert f_audio.shape == (2, 16, 32)
        assert f_seismic.shape == (2, 16, 32)
        # Concat and feed encoder.
        feat = torch.cat([f_audio, f_seismic], dim=-1)  # (B, C, 64)
        with torch.no_grad():
            z, mu, logvar = encoder(feat)
        assert mu.shape == (2, cfg.d_z)
