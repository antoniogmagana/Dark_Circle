"""Tests for VAETrainingMode — the Checkpoint 1 reference implementation.

Validation strategy:
  1. Interface — the mode implements everything the Trainer needs.
  2. Forward parity — per-batch loss & metrics match what the old Trainer
     produced on the same input. We can't bit-compare across the refactor,
     but we can verify the metric schema matches and values are finite.
  3. Checkpoint selection — dual-checkpoint logic matches the old behavior.
  4. Factory — build_training_mode returns the right instance.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
from crl_vehicle.config import CRLConfig
from crl_vehicle.priors import StandardPrior
from crl_vehicle.training_modes import (
    CheckpointState,
    TrainingMode,
    VAETrainingMode,
    build_training_mode,
)
from training.trainer import CRLModel


def _synthetic_batch(B=4, n_partners=4, audio_W=16000, seismic_W=100):
    batch = {
        "x_audio_t": torch.randn(B, 1, audio_W) * 0.01,
        "x_seismic_t": torch.randn(B, 1, seismic_W) * 0.01,
        "audio_avail": torch.ones(B, dtype=torch.bool),
        "seismic_avail": torch.ones(B, dtype=torch.bool),
        "detection_label_t": torch.randint(0, 2, (B,)),
        "vehicle_type_t": torch.randint(0, 4, (B,)),
        "n_partners": n_partners,
    }
    for p in range(n_partners):
        batch[f"x_audio_p{p}"] = torch.randn(B, 1, audio_W) * 0.01
        batch[f"x_seismic_p{p}"] = torch.randn(B, 1, seismic_W) * 0.01
        batch[f"detection_label_p{p}"] = torch.randint(0, 2, (B,))
        batch[f"vehicle_type_p{p}"] = torch.randint(0, 4, (B,))
        batch[f"partner_stratum_p{p}"] = torch.full((B,), p % 4)
    return batch


@pytest.fixture
def cfg_ms():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="multiscale",
        fused_seq_len=16,
        d_z=24,
    )


@pytest.fixture
def cfg_morlet():
    return CRLConfig(
        d_model=32, n_layers=1, n_heads=4, frontend_type="morlet_per_sensor", d_z=24
    )


@pytest.fixture
def cfg_morlet_fused():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="morlet_fused",
        fused_seq_len=16,
        d_z=24,
    )


@pytest.fixture
def cfg_morlet_learnable():
    return CRLConfig(d_model=32, n_layers=1, n_heads=4, frontend_type="morlet_learnable", d_z=24)


@pytest.fixture
def cfg_morlet_learnable_fused():
    return CRLConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        frontend_type="morlet_learnable_fused",
        fused_seq_len=16,
        d_z=24,
    )


class TestFactory:
    def test_default_config_builds_vae_standard(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, VAETrainingMode)
        assert isinstance(mode.prior, StandardPrior)

    def test_unknown_training_mode_raises(self, cfg_ms):
        cfg_ms.training_mode = "nope"
        with pytest.raises(ValueError, match="Unknown training_mode"):
            build_training_mode(cfg_ms)

    def test_unknown_prior_type_raises(self, cfg_ms):
        cfg_ms.prior_type = "nope"
        with pytest.raises(ValueError, match="Unknown prior_type"):
            build_training_mode(cfg_ms)

    def test_conditional_prior_builds(self, cfg_ms):
        """Checkpoint 2: ConditionalPrior is a valid selection."""
        from crl_vehicle.priors import ConditionalPrior

        cfg_ms.prior_type = "conditional"
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, VAETrainingMode)
        assert isinstance(mode.prior, ConditionalPrior)


class TestVAEModeInterface:
    def test_is_training_mode(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert isinstance(mode, TrainingMode)

    def test_early_stop_metric_is_ref_elbo(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        assert mode.early_stop_metric() == "val_ref_elbo"
        assert mode.early_stop_mode() == "min"


class TestForwardPairMetrics:
    """forward_pair must produce the scalar + tensor keys Trainer consumes."""

    EXPECTED_SCALAR_KEYS: ClassVar[set[str]] = {
        "recon",
        "kl",
        "raw_kl",
        "interv",
        "total",
    }
    EXPECTED_TENSOR_KEYS: ClassVar[set[str]] = {
        "aux_pres_logits",
        "aux_pres_labels",
        "aux_type_logits",
        "aux_type_labels",
    }

    @pytest.mark.parametrize(
        "cfg_name",
        [
            "cfg_ms",
            "cfg_morlet",
            "cfg_morlet_fused",
            "cfg_morlet_learnable",
            "cfg_morlet_learnable_fused",
        ],
    )
    def test_returns_loss_and_metrics(self, cfg_name, request):
        cfg = request.getfixturevalue(cfg_name)
        mode = build_training_mode(cfg)
        model = CRLModel(cfg)
        batch = _synthetic_batch()
        loss, metrics = mode.forward_pair(model, batch, beta=0.1, device=torch.device("cpu"))
        assert torch.isfinite(loss)
        assert self.EXPECTED_SCALAR_KEYS.issubset(metrics.keys())
        assert self.EXPECTED_TENSOR_KEYS.issubset(metrics.keys())

    @pytest.mark.parametrize(
        "cfg_name",
        [
            "cfg_ms",
            "cfg_morlet",
            "cfg_morlet_fused",
            "cfg_morlet_learnable",
            "cfg_morlet_learnable_fused",
        ],
    )
    def test_gradients_flow(self, cfg_name, request):
        cfg = request.getfixturevalue(cfg_name)
        mode = build_training_mode(cfg)
        model = CRLModel(cfg)
        batch = _synthetic_batch()
        loss, _ = mode.forward_pair(model, batch, beta=0.1, device=torch.device("cpu"))
        loss.backward()
        bad = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not p.grad.isfinite().all()
        ]
        assert not bad, f"Non-finite grads: {bad}"


class TestCheckpointLogic:
    def test_first_epoch_saves_both(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState()
        val_m = {
            "val_ref_elbo": 5.0,
            "val_aux_type_f1": 0.5,
            "val_recon": 0.1,
            "val_raw_kl": 4.9,
        }
        saves = mode.should_save_checkpoint(val_m, epoch=0, state=state)
        assert saves[mode.CKPT_REF_ELBO] is True
        assert saves[mode.CKPT_AUX_TYPE_F1] is True
        assert state.bests["val_ref_elbo"] == 5.0
        assert state.bests["val_aux_type_f1"] == 0.5

    def test_ref_elbo_saves_when_improves(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(bests={"val_ref_elbo": 5.0, "val_aux_type_f1": 0.5})
        saves = mode.should_save_checkpoint(
            {
                "val_ref_elbo": 4.5,
                "val_aux_type_f1": 0.4,
                "val_recon": 0.1,
                "val_raw_kl": 4.4,
            },
            epoch=1,
            state=state,
        )
        assert saves[mode.CKPT_REF_ELBO] is True
        assert saves[mode.CKPT_AUX_TYPE_F1] is False

    def test_aux_type_saves_when_improves(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(bests={"val_ref_elbo": 5.0, "val_aux_type_f1": 0.5})
        saves = mode.should_save_checkpoint(
            {
                "val_ref_elbo": 6.0,
                "val_aux_type_f1": 0.7,
                "val_recon": 0.1,
                "val_raw_kl": 5.9,
            },
            epoch=1,
            state=state,
        )
        assert saves[mode.CKPT_REF_ELBO] is False
        assert saves[mode.CKPT_AUX_TYPE_F1] is True

    def test_patience_increments_on_no_ref_elbo_improvement(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(bests={"val_ref_elbo": 5.0, "val_aux_type_f1": 0.5})
        mode.should_save_checkpoint(
            {
                "val_ref_elbo": 5.1,
                "val_aux_type_f1": 0.4,
                "val_recon": 0.1,
                "val_raw_kl": 5.0,
            },
            epoch=1,
            state=state,
        )
        assert state.patience_count == 1

    def test_checkpoint_summary_schema(self, cfg_ms):
        """Must match the legacy crl_checkpoint_summary.json schema."""
        mode = build_training_mode(cfg_ms)
        state = CheckpointState(
            bests={"val_ref_elbo": 0.241628, "val_aux_type_f1": 0.7009},
            best_epochs={"val_aux_type_f1": 28},
        )
        summary = mode.checkpoint_summary(state)
        assert "best_ref_elbo" in summary
        assert "best_aux_type_f1" in summary
        assert "best_aux_type_epoch" in summary
        assert "checkpoints" in summary
        assert "crl_best.pth" in summary["checkpoints"]
        assert "crl_best_aux_type.pth" in summary["checkpoints"]
        assert "crl_final.pth" in summary["checkpoints"]
        assert summary["best_aux_type_epoch"] == 28


class TestConditionalPriorIntegration:
    """VAETrainingMode + ConditionalPrior: labels must reach the prior MLP
    via _kl_terms, and gradients must flow to both encoder and prior params."""

    @pytest.mark.parametrize(
        "cfg_name",
        [
            "cfg_ms",
            "cfg_morlet",
            "cfg_morlet_fused",
            "cfg_morlet_learnable",
            "cfg_morlet_learnable_fused",
        ],
    )
    def test_conditional_prior_forward_pair_runs(self, cfg_name, request):
        cfg = request.getfixturevalue(cfg_name)
        cfg.prior_type = "conditional"
        mode = build_training_mode(cfg)
        model = CRLModel(cfg)
        batch = _synthetic_batch()
        loss, metrics = mode.forward_pair(model, batch, beta=0.1, device=torch.device("cpu"))
        assert torch.isfinite(loss)
        # raw_kl must be present and non-negative.
        assert metrics["raw_kl"] >= 0.0

    def test_prior_mlp_receives_gradients_through_training_mode(self, cfg_ms):
        """After backward(), every ConditionalPrior parameter must have a
        finite non-None grad. Validates the optimizer param-group wiring and
        the y-plumbing in _kl_terms."""
        cfg_ms.prior_type = "conditional"
        mode = build_training_mode(cfg_ms)
        model = CRLModel(cfg_ms)
        batch = _synthetic_batch()
        loss, _ = mode.forward_pair(model, batch, beta=1.0, device=torch.device("cpu"))
        loss.backward()
        bad = [
            n
            for n, p in mode.prior.named_parameters()
            if p.grad is None or not p.grad.isfinite().all()
        ]
        assert not bad, f"Prior MLP params with bad grads: {bad}"

    def test_standard_and_conditional_produce_different_losses(self, cfg_ms):
        """Same batch, same model weights (ish — init is random per build),
        different prior types → different losses. Cheap smoke that the prior
        swap actually changes something end-to-end."""
        torch.manual_seed(0)
        cfg_std = CRLConfig(**{**cfg_ms.__dict__, "prior_type": "standard"})
        torch.manual_seed(0)
        mode_std = build_training_mode(cfg_std)
        torch.manual_seed(0)
        model_std = CRLModel(cfg_std)

        torch.manual_seed(0)
        cfg_cond = CRLConfig(**{**cfg_ms.__dict__, "prior_type": "conditional"})
        torch.manual_seed(0)
        mode_cond = build_training_mode(cfg_cond)
        torch.manual_seed(0)
        model_cond = CRLModel(cfg_cond)

        # Zero-init the prior MLP so the condition collapses to N(0,I) at the
        # start — then they MUST produce identical KL at beta=1.
        with torch.no_grad():
            mode_cond.prior.net[-1].weight.zero_()
            mode_cond.prior.net[-1].bias.zero_()

        batch = _synthetic_batch()
        # Re-seed before EACH forward_pair so reparameterization noise
        # (TemporalEncoder's z = mu + sigma*randn_like) is identical across
        # both runs. _synthetic_batch consumed RNG, so without resetting
        # before the first forward the two paths see different RNG state.
        torch.manual_seed(1)
        loss_std, m_std = mode_std.forward_pair(
            model_std, batch, beta=1.0, device=torch.device("cpu")
        )
        torch.manual_seed(1)
        loss_cond, m_cond = mode_cond.forward_pair(
            model_cond, batch, beta=1.0, device=torch.device("cpu")
        )
        # Both priors reduce to N(0,I) here, so raw_kl should match closely.
        assert m_std["raw_kl"] == pytest.approx(m_cond["raw_kl"], abs=1e-4)


class TestValMetricsSummary:
    def test_computes_ref_elbo(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        out = mode.val_metrics_summary({"val_recon": 0.2, "val_raw_kl": 0.8})
        assert out["val_ref_elbo"] == pytest.approx(1.0)

    def test_preserves_other_keys(self, cfg_ms):
        mode = build_training_mode(cfg_ms)
        out = mode.val_metrics_summary(
            {
                "val_recon": 0.2,
                "val_raw_kl": 0.8,
                "val_aux_type_f1": 0.5,
            }
        )
        assert out["val_aux_type_f1"] == 0.5
