import pytest
import torch
import torch.nn.functional as F
from crl_vehicle.losses.crl_loss import (
    reconstruction_loss, kl_divergence, intervention_matching_loss
)


class TestReconstructionLoss:

    def test_zero_on_identical(self):
        x = torch.randn(4, 64, 32)
        assert reconstruction_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_different(self):
        assert reconstruction_loss(torch.randn(4, 64, 32), torch.randn(4, 64, 32)).item() > 0.0

    def test_scalar(self):
        assert reconstruction_loss(torch.randn(4, 64, 32), torch.randn(4, 64, 32)).shape == ()


class TestKLDivergence:

    def test_zero_at_prior(self):
        assert kl_divergence(torch.zeros(8, 24), torch.zeros(8, 24)).item() == pytest.approx(0.0, abs=1e-5)

    def test_beta_scaling(self):
        mu, lv = torch.ones(4, 24), torch.zeros(4, 24)
        assert kl_divergence(mu, lv, beta=2.0).item() == pytest.approx(
            2.0 * kl_divergence(mu, lv, beta=1.0).item(), rel=1e-4
        )

    def test_sums_over_dims(self):
        # KL(N(1,1)||N(0,1)) per dim = 0.5*(1+1-1-0)=0.5, total=24*0.5=12.0
        mu, lv = torch.ones(4, 24), torch.zeros(4, 24)
        assert kl_divergence(mu, lv, beta=1.0).item() == pytest.approx(12.0, rel=1e-4)

    def test_scalar(self):
        assert kl_divergence(torch.randn(4, 24), torch.zeros(4, 24)).shape == ()


class TestInterventionMatchingLoss:

    def test_scalar(self):
        assert intervention_matching_loss(
            torch.randn(8, 2), torch.randint(0, 2, (8, 2)).float()
        ).shape == ()

    def test_matches_bce_with_logits(self):
        logits  = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8, 2)).float()
        expected = F.binary_cross_entropy_with_logits(logits, targets)
        assert intervention_matching_loss(logits, targets).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_low_for_perfect_prediction(self):
        targets = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        logits  = targets * 10.0 - (1.0 - targets) * 10.0
        assert intervention_matching_loss(logits, targets).item() < 0.01

    def test_finite(self):
        assert intervention_matching_loss(torch.randn(16, 2), torch.zeros(16, 2)).isfinite()
