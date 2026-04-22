import pytest
import torch

from crl_vehicle.losses.crl_loss import kl_divergence
from crl_vehicle.priors import Prior, StandardPrior


class TestStandardPrior:
    def test_is_prior_subclass(self):
        p = StandardPrior()
        assert isinstance(p, Prior)

    def test_zero_mu_zero_logvar_kl_is_zero(self):
        """KL[N(0, I) || N(0, I)] = 0."""
        p = StandardPrior()
        mu = torch.zeros(8, 24)
        logvar = torch.zeros(8, 24)
        kl = p.kl_to_posterior(mu, logvar)
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_nonneg(self):
        p = StandardPrior()
        mu = torch.randn(16, 12)
        logvar = torch.randn(16, 12).clamp(-4, 4)
        kl = p.kl_to_posterior(mu, logvar)
        assert kl.item() >= 0.0

    def test_matches_legacy_kl_divergence_at_beta_one(self):
        """Must produce the same scalar as losses.crl_loss.kl_divergence(beta=1.0)."""
        torch.manual_seed(0)
        mu = torch.randn(32, 24)
        logvar = torch.randn(32, 24).clamp(-4, 4)
        p = StandardPrior()
        expected = kl_divergence(mu, logvar, beta=1.0)
        got = p.kl_to_posterior(mu, logvar)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_ignores_y(self):
        """Passing y must not change output (y is only used by conditional priors)."""
        p = StandardPrior()
        torch.manual_seed(0)
        mu = torch.randn(4, 8)
        logvar = torch.randn(4, 8).clamp(-4, 4)
        kl_no_y = p.kl_to_posterior(mu, logvar, y=None)
        kl_y = p.kl_to_posterior(mu, logvar, y=torch.randn(4, 2))
        assert torch.equal(kl_no_y, kl_y)

    def test_backprop_flows(self):
        p = StandardPrior()
        mu = torch.randn(4, 8, requires_grad=True)
        logvar = torch.randn(4, 8, requires_grad=True)
        loss = p.kl_to_posterior(mu, logvar)
        loss.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert logvar.grad is not None and torch.isfinite(logvar.grad).all()
