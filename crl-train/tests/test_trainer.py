import pytest
import torch
import torch.nn as nn
from crl_vehicle.config import CRLConfig
from training.trainer import CRLModel, Trainer


@pytest.fixture
def cfg_ms():
    return CRLConfig(d_model=32, n_layers=1, n_heads=4,
                     frontend_type="multiscale", fused_seq_len=16, d_z=24)


@pytest.fixture
def cfg_morlet():
    return CRLConfig(d_model=32, n_layers=1, n_heads=4,
                     frontend_type="morlet", d_z=24)


def _synthetic_batch(B=4, n_partners=4):
    batch = {
        "x_audio_t":         torch.randn(B, 1, 16000) * 0.01,
        "x_seismic_t":       torch.randn(B, 1, 200) * 0.01,
        "audio_avail":       torch.ones(B, dtype=torch.bool),
        "seismic_avail":     torch.ones(B, dtype=torch.bool),
        "detection_label_t": torch.randint(0, 2, (B,)),
        "vehicle_type_t":    torch.randint(0, 4, (B,)),
        "n_partners": n_partners,
    }
    for p in range(n_partners):
        batch[f"x_audio_p{p}"]         = torch.randn(B, 1, 16000) * 0.01
        batch[f"x_seismic_p{p}"]       = torch.randn(B, 1, 200) * 0.01
        batch[f"detection_label_p{p}"] = torch.randint(0, 2, (B,))
        batch[f"vehicle_type_p{p}"]    = torch.randint(0, 4, (B,))
        batch[f"partner_stratum_p{p}"] = torch.full((B,), p % 4)
    return batch


class TestCRLModelMultiscale:

    def test_shared_encoder_exists(self, cfg_ms):
        model = CRLModel(cfg_ms)
        assert hasattr(model, "encoder") and model.encoder is not None

    def test_shared_decoder_exists(self, cfg_ms):
        model = CRLModel(cfg_ms)
        assert hasattr(model, "decoder") and model.decoder is not None

    def test_frontends_have_adaptive_pool(self, cfg_ms):
        model = CRLModel(cfg_ms)
        for sensor in ["audio", "seismic"]:
            assert any(isinstance(m, nn.AdaptiveAvgPool1d)
                       for m in model.frontends[sensor].modules()), \
                f"{sensor} frontend missing AdaptiveAvgPool1d"

    def test_encode_fused_output_shape(self, cfg_ms):
        model = CRLModel(cfg_ms)
        model.eval()
        T = cfg_ms.fused_seq_len
        with torch.no_grad():
            features, z, mu, logvar = model.encode_fused(
                torch.zeros(4, 1, 16000), torch.zeros(4, 1, 200)
            )
        assert features.shape == (4, cfg_ms.d_model, 2 * T)
        assert z.shape == (4, 24)
        assert mu.shape == (4, 24)

    def test_decode_fused_shape_matches_features(self, cfg_ms):
        model = CRLModel(cfg_ms)
        model.eval()
        with torch.no_grad():
            features, z, _, _ = model.encode_fused(
                torch.zeros(4, 1, 16000), torch.zeros(4, 1, 200)
            )
            x_hat = model.decode_fused(z)
        assert x_hat.shape == features.shape

    def test_encode_fused_finite(self, cfg_ms):
        model = CRLModel(cfg_ms)
        model.eval()
        with torch.no_grad():
            features, z, mu, _ = model.encode_fused(
                torch.randn(4, 1, 16000) * 0.01, torch.randn(4, 1, 200) * 0.01
            )
        assert features.isfinite().all() and z.isfinite().all() and mu.isfinite().all()


class TestCRLModelMorlet:

    def test_per_sensor_encoders(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        assert "audio" in model.encoders and "seismic" in model.encoders

    def test_per_sensor_decoders(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        assert "audio" in model.decoders and "seismic" in model.decoders

    def test_encode_audio_shape(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        model.eval()
        with torch.no_grad():
            _, z, _, _ = model.encode("audio", torch.zeros(4, 1, 16000))
        assert z.shape == (4, 24)

    def test_encode_seismic_shape(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        model.eval()
        with torch.no_grad():
            _, z, _, _ = model.encode("seismic", torch.zeros(4, 1, 200))
        assert z.shape == (4, 24)

    def test_no_shared_encoder(self, cfg_morlet):
        model = CRLModel(cfg_morlet)
        assert not hasattr(model, "encoder") or model.encoder is None


class TestCRLModelShared:

    @pytest.mark.parametrize("fe", ["multiscale", "morlet", "morlet_per_sensor", "morlet_fused", "morlet_learnable", "morlet_learnable_fused"])
    def test_has_latent(self, fe):
        from crl_vehicle.models.latent import CausalLatentSpace
        model = CRLModel(CRLConfig(frontend_type=fe, d_model=32, n_layers=1))
        assert isinstance(model.latent, CausalLatentSpace)

    @pytest.mark.parametrize("fe", ["multiscale", "morlet", "morlet_per_sensor", "morlet_fused", "morlet_learnable", "morlet_learnable_fused"])
    def test_has_interv_classifier(self, fe):
        model = CRLModel(CRLConfig(frontend_type=fe, d_model=32, n_layers=1))
        assert hasattr(model, "interv_classifier")

    @pytest.mark.parametrize("fe", ["multiscale", "morlet", "morlet_per_sensor", "morlet_fused", "morlet_learnable", "morlet_learnable_fused"])
    def test_backbone_excludes_downstream_heads(self, fe):
        model = CRLModel(CRLConfig(frontend_type=fe, d_model=32, n_layers=1))
        backbone_ids = set(id(p) for p in model.backbone_parameters())
        for hd in [model.pres_heads, model.type_heads, model.prox_heads]:
            for p in hd.parameters():
                assert id(p) not in backbone_ids

    def test_unknown_frontend_raises(self):
        with pytest.raises(ValueError, match="Unknown frontend_type"):
            CRLModel(CRLConfig(frontend_type="bad", d_model=32, n_layers=1))


class TestForwardPair:

    @pytest.mark.parametrize("fe", ["multiscale", "morlet", "morlet_per_sensor", "morlet_fused", "morlet_learnable", "morlet_learnable_fused"])
    def test_finite_loss(self, fe, tmp_path):
        cfg = CRLConfig(frontend_type=fe, d_model=32, n_layers=1, fused_seq_len=16)
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        model.train()
        loss, metrics = trainer._forward_pair(_synthetic_batch(), beta=0.5)
        assert loss.isfinite(), f"Loss not finite: {loss.item()}"

    @pytest.mark.parametrize("fe", ["multiscale", "morlet", "morlet_per_sensor", "morlet_fused", "morlet_learnable", "morlet_learnable_fused"])
    def test_metrics_keys(self, fe, tmp_path):
        cfg = CRLConfig(frontend_type=fe, d_model=32, n_layers=1, fused_seq_len=16)
        trainer = Trainer(CRLModel(cfg), cfg, torch.device("cpu"), tmp_path)
        _, metrics = trainer._forward_pair(_synthetic_batch(), beta=0.5)
        for k in ("recon", "kl", "raw_kl", "interv", "total"):
            assert k in metrics

    @pytest.mark.parametrize("fe", ["multiscale", "morlet", "morlet_per_sensor", "morlet_fused", "morlet_learnable", "morlet_learnable_fused"])
    def test_finite_grads(self, fe, tmp_path):
        cfg = CRLConfig(frontend_type=fe, d_model=32, n_layers=1, fused_seq_len=16)
        model = CRLModel(cfg)
        trainer = Trainer(model, cfg, torch.device("cpu"), tmp_path)
        model.train()
        trainer.optimizer.zero_grad()
        loss, _ = trainer._forward_pair(_synthetic_batch(), beta=0.5)
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not p.grad.isfinite().all()]
        assert not bad, f"Non-finite grads: {bad}"


class TestBetaAnnealing:
    """Beta schedule lives in VAETrainingMode.update_beta — Trainer only
    stores self.beta and passes it through. These tests drive the mode
    directly, bypassing the epoch loop."""

    @pytest.fixture
    def trainer(self, tmp_path):
        cfg = CRLConfig(d_model=32, n_layers=1, frontend_type="morlet",
                        kl_floor=0.01, kl_target=0.5, beta_step=0.1)
        return Trainer(CRLModel(cfg), cfg, torch.device("cpu"), tmp_path)

    def _step(self, trainer, val_m):
        """Invoke mode.update_beta with trainer's current state and assign."""
        new_beta, event = trainer.mode.update_beta(
            trainer.beta, val_m, trainer.ckpt_state, trainer.cfg
        )
        trainer.beta = new_beta
        return event

    def test_initial_beta_zero(self, trainer):
        assert trainer.beta == 0.0

    def test_mode_owns_update_beta(self, trainer):
        assert hasattr(trainer.mode, "update_beta")

    def test_increases_when_kl_above_target(self, trainer):
        trainer.ckpt_state.prev_val_recon = 1.0
        self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 1.0})
        assert trainer.beta > 0.0

    def test_decreases_on_collapse(self, trainer):
        trainer.beta = 0.5
        trainer.ckpt_state.prev_val_recon = 1.0
        self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 0.001})
        assert trainer.beta < 0.5

    def test_bounded_0_to_1(self, trainer):
        trainer.beta = 1.0
        trainer.ckpt_state.prev_val_recon = 1.0
        self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 1.0})
        assert trainer.beta <= 1.0

    def test_returns_event_string(self, trainer):
        trainer.ckpt_state.prev_val_recon = 1.0
        event = self._step(trainer, {"val_recon": 0.9, "val_raw_kl": 0.8})
        assert event in ("↑", "→hold", "↓collapse")


class TestLinearSignalProbe:
    """linear_signal builds a Linear(d_signal, 4) head; eval slices z[:d_signal]."""

    def test_builds_head_sized_to_d_signal(self):
        cfg = CRLConfig(d_model=32, n_layers=1, n_heads=4,
                        frontend_type="multiscale", fused_seq_len=16,
                        d_z=24, d_signal=12)
        model = CRLModel(cfg, probe_mode="linear_signal")
        head = model.type_heads["fused"].head
        assert head.weight.shape == (4, 12)

    def test_d_signal_16_makes_16_dim_head(self):
        cfg = CRLConfig(d_model=32, n_layers=1, n_heads=4,
                        frontend_type="multiscale", fused_seq_len=16,
                        d_z=24, d_signal=16)
        model = CRLModel(cfg, probe_mode="linear_signal")
        head = model.type_heads["fused"].head
        assert head.weight.shape == (4, 16)

    def test_per_sensor_frontend_also_works(self):
        cfg = CRLConfig(d_model=32, n_layers=1, n_heads=4,
                        frontend_type="morlet_per_sensor", fused_seq_len=16,
                        d_z=24, d_signal=12)
        model = CRLModel(cfg, probe_mode="linear_signal")
        for sensor in ("audio", "seismic"):
            head = model.type_heads[sensor].head
            assert head.weight.shape == (4, 12)

    def test_invalid_probe_mode_still_rejected(self):
        cfg = CRLConfig(d_model=32, n_layers=1, n_heads=4,
                        frontend_type="multiscale", fused_seq_len=16, d_z=24)
        with pytest.raises(ValueError, match="probe_mode must be one of"):
            CRLModel(cfg, probe_mode="bogus")

    def test_valid_probe_modes_includes_linear_signal(self):
        assert "linear_signal" in CRLModel.VALID_PROBE_MODES
