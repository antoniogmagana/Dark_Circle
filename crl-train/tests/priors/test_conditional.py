import pytest
import torch

from crl_vehicle.priors import ConditionalPrior, Prior
from crl_vehicle.priors.conditional import _encode_labels, _LABEL_DIM


def _y(presence: list, types: list) -> torch.Tensor:
    """Build (B, 2) label tensor. types is raw {-2, -1, 0, 1, 2, 3}."""
    assert len(presence) == len(types)
    return torch.tensor(
        [[float(p), float(t)] for p, t in zip(presence, types)]
    )


class TestLabelEncoding:
    def test_shape(self):
        y = _y([1, 0, 1], [2, -1, 3])
        enc = _encode_labels(y)
        assert enc.shape == (3, _LABEL_DIM)

    def test_presence_bit_preserved(self):
        y = _y([1, 0], [0, 0])
        enc = _encode_labels(y)
        assert enc[0, 0].item() == 1.0
        assert enc[1, 0].item() == 0.0

    def test_type_onehot_sums_to_one(self):
        y = _y([1, 1, 1, 1, 1, 1], [-2, -1, 0, 1, 2, 3])
        enc = _encode_labels(y)
        # columns 1..end are the onehot block
        assert torch.allclose(enc[:, 1:].sum(dim=-1), torch.ones(6))

    def test_distinct_labels_distinct_encoding(self):
        y = _y([1, 1], [0, 3])
        enc = _encode_labels(y)
        # Rows must differ in the onehot block.
        assert not torch.equal(enc[0, 1:], enc[1, 1:])


class TestConditionalPriorBasics:
    def test_is_prior_subclass(self):
        p = ConditionalPrior(d_z=24)
        assert isinstance(p, Prior)

    def test_prior_params_shape(self):
        p = ConditionalPrior(d_z=24)
        y = _y([1, 0, 1], [2, -1, 3])
        mu_p, logvar_p = p.prior_params(y)
        assert mu_p.shape == (3, 24)
        assert logvar_p.shape == (3, 24)

    def test_small_init_produces_near_standard_prior(self):
        """init_scale=0.01 default: prior ≈ N(0, I) at t=0."""
        torch.manual_seed(0)
        p = ConditionalPrior(d_z=24, init_scale=0.01)
        y = _y([1, 0, 1, 1], [2, -1, 3, 0])
        mu_p, logvar_p = p.prior_params(y)
        # μ close to 0, logvar close to 0 (i.e., σ close to 1).
        assert mu_p.abs().max().item() < 0.1
        assert logvar_p.abs().max().item() < 0.1

    def test_kl_requires_labels(self):
        p = ConditionalPrior(d_z=8)
        with pytest.raises(ValueError, match="requires auxiliary labels"):
            p.kl_to_posterior(torch.zeros(2, 8), torch.zeros(2, 8), y=None)

    def test_logvar_clamped(self):
        """logvar clamped to [LOGVAR_MIN, LOGVAR_MAX] — prior variance bounded."""
        torch.manual_seed(0)
        # Use large init_scale so the network can produce extreme values.
        p = ConditionalPrior(d_z=8, init_scale=100.0)
        y = _y([1] * 32, [0, 1, 2, 3, -1, -2, 0, 1] * 4)
        _, logvar_p = p.prior_params(y)
        assert logvar_p.min().item() >= p.LOGVAR_MIN - 1e-6
        assert logvar_p.max().item() <= p.LOGVAR_MAX + 1e-6


class TestConditionalPriorKL:
    def test_kl_zero_when_posterior_matches_prior(self):
        """KL[q || p] = 0 when q and p have identical μ, σ.

        Set init_scale=0 → prior is exactly N(0, I). Pass mu=0, logvar=0
        as posterior → KL must be ~0.
        """
        p = ConditionalPrior(d_z=8, init_scale=0.0)
        y = _y([1, 0], [2, 0])
        mu = torch.zeros(2, 8)
        logvar = torch.zeros(2, 8)
        kl = p.kl_to_posterior(mu, logvar, y=y)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_nonneg(self):
        torch.manual_seed(1)
        p = ConditionalPrior(d_z=12)
        y = _y([1, 0, 1, 0], [2, 0, 3, -1])
        mu = torch.randn(4, 12)
        logvar = torch.randn(4, 12).clamp(-4, 4)
        kl = p.kl_to_posterior(mu, logvar, y=y)
        assert kl.item() >= 0.0

    def test_kl_matches_standard_when_prior_is_standard(self):
        """With init_scale=0, ConditionalPrior produces prior=N(0,I) for every
        label. KL[q || N(0, I)] must then match StandardPrior's KL exactly."""
        from crl_vehicle.priors import StandardPrior

        torch.manual_seed(2)
        p = ConditionalPrior(d_z=16, init_scale=0.0)
        sp = StandardPrior()
        y = _y([1, 0, 1, 0], [2, 0, 3, -1])
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16).clamp(-4, 4)
        kl_cond = p.kl_to_posterior(mu, logvar, y=y)
        kl_std  = sp.kl_to_posterior(mu, logvar, y=y)
        assert torch.allclose(kl_cond, kl_std, atol=1e-5)

    def test_kl_differs_between_labels_after_training(self):
        """After giving the prior MLP nonzero weights, different labels
        should produce different priors → different KL values."""
        torch.manual_seed(3)
        # Use a non-zero init so prior varies across labels.
        p = ConditionalPrior(d_z=8, init_scale=1.0)
        mu = torch.randn(2, 8)
        logvar = torch.zeros(2, 8)

        y_a = _y([1], [0])  # pedestrian
        y_b = _y([1], [3])  # heavy
        kl_a = p.kl_to_posterior(mu[:1], logvar[:1], y=y_a)
        kl_b = p.kl_to_posterior(mu[:1], logvar[:1], y=y_b)
        assert kl_a.item() != kl_b.item()


class TestConditionalPriorTraining:
    def test_gradients_flow_into_prior_mlp(self):
        """KL backward must give finite grads to the prior MLP — otherwise
        iVAE can't adapt the prior to labels."""
        p = ConditionalPrior(d_z=8)
        y = _y([1, 0], [2, 0])
        mu = torch.randn(2, 8, requires_grad=True)
        logvar = torch.zeros(2, 8, requires_grad=True)
        kl = p.kl_to_posterior(mu, logvar, y=y)
        kl.backward()
        bad = [n for n, prm in p.named_parameters()
               if prm.grad is not None and not prm.grad.isfinite().all()]
        assert not bad, f"Non-finite prior-MLP grads: {bad}"

    def test_gradients_flow_into_posterior(self):
        """KL must also give finite grads to (mu, logvar) so the encoder learns."""
        p = ConditionalPrior(d_z=8)
        y = _y([1, 0], [2, 0])
        mu = torch.randn(2, 8, requires_grad=True)
        logvar = torch.zeros(2, 8, requires_grad=True)
        kl = p.kl_to_posterior(mu, logvar, y=y)
        kl.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert logvar.grad is not None and torch.isfinite(logvar.grad).all()

    def test_mlp_params_included_in_optimizer_discovery(self):
        """ConditionalPrior is an nn.Module — its parameters must be discoverable
        by the Trainer's optimizer. Checkpoint 1 set up the param-group logic."""
        p = ConditionalPrior(d_z=8)
        params = list(p.parameters())
        assert len(params) > 0
        assert all(prm.requires_grad for prm in params)
