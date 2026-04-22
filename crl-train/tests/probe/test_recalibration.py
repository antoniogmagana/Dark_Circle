import math

import pytest
import torch

from crl_vehicle.probe.recalibration import (
    UNIFORM_BINARY_PRIOR,
    apply_binary_log_prior_shift,
    apply_multiclass_log_prior_shift,
    compute_binary_prior,
    compute_multiclass_prior,
    uniform_multiclass_prior,
)


class TestPriorComputation:
    def test_uniform_binary_prior_constant(self):
        assert UNIFORM_BINARY_PRIOR == 0.5

    def test_uniform_multiclass_prior_sums_to_one(self):
        p = uniform_multiclass_prior(4)
        assert p.shape == (4,)
        assert torch.allclose(p.sum(), torch.tensor(1.0))
        assert torch.allclose(p, torch.full((4,), 0.25))

    def test_binary_prior_balanced(self):
        labels = torch.tensor([0, 1, 0, 1])
        assert compute_binary_prior(labels) == pytest.approx(0.5)

    def test_binary_prior_imbalanced(self):
        labels = torch.tensor([1, 1, 1, 0])
        assert compute_binary_prior(labels) == pytest.approx(0.75)

    def test_binary_prior_empty_falls_back_to_uniform(self):
        assert compute_binary_prior(torch.empty(0)) == UNIFORM_BINARY_PRIOR

    def test_binary_prior_clamped_away_from_0_and_1(self):
        assert compute_binary_prior(torch.zeros(10)) > 0.0
        assert compute_binary_prior(torch.ones(10)) < 1.0

    def test_multiclass_prior_matches_bincount(self):
        labels = torch.tensor([0, 0, 1, 2, 3, 3, 3])
        p = compute_multiclass_prior(labels, n_classes=4)
        # 2/7, 1/7, 1/7, 3/7
        expected = torch.tensor([2, 1, 1, 3]) / 7.0
        assert torch.allclose(p, expected, atol=1e-6)

    def test_multiclass_prior_empty_falls_back_to_uniform(self):
        p = compute_multiclass_prior(torch.empty(0, dtype=torch.long), n_classes=4)
        assert torch.allclose(p, uniform_multiclass_prior(4))


class TestBinaryLogPriorShift:
    def test_uniform_split_uniform_train_zero_shift(self):
        logits = torch.randn(10)
        shifted = apply_binary_log_prior_shift(logits, p_split=0.5, p_train=0.5)
        assert torch.allclose(shifted, logits)

    def test_shift_matches_log_odds_difference(self):
        logit = torch.tensor([0.0])
        p_split, p_train = 0.9, 0.5
        expected = math.log(0.9 / 0.1) - math.log(0.5 / 0.5)
        shifted = apply_binary_log_prior_shift(logit, p_split, p_train)
        assert shifted.item() == pytest.approx(expected)

    def test_positive_class_heavier_in_split_biases_toward_positive(self):
        # Raw logit of 0 is the decision boundary. If split is heavily positive,
        # the shifted logit at 0 should become positive (i.e. predict class 1).
        shifted = apply_binary_log_prior_shift(
            torch.tensor([0.0]), p_split=0.9, p_train=0.5
        )
        assert shifted.item() > 0

    def test_negative_class_heavier_in_split_biases_toward_negative(self):
        shifted = apply_binary_log_prior_shift(
            torch.tensor([0.0]), p_split=0.1, p_train=0.5
        )
        assert shifted.item() < 0


class TestMulticlassLogPriorShift:
    def test_uniform_split_uniform_train_zero_shift(self):
        logits = torch.randn(5, 4)
        shifted = apply_multiclass_log_prior_shift(
            logits, p_split=uniform_multiclass_prior(4)
        )
        assert torch.allclose(shifted, logits, atol=1e-6)

    def test_shift_matches_formula(self):
        logits = torch.zeros(1, 4)
        p_split = torch.tensor([0.4, 0.3, 0.2, 0.1])
        p_train = uniform_multiclass_prior(4)
        shifted = apply_multiclass_log_prior_shift(logits, p_split, p_train)
        expected = torch.log(p_split) - torch.log(p_train)
        assert torch.allclose(shifted[0], expected, atol=1e-6)

    def test_argmax_moves_toward_dominant_split_class_under_equal_logits(self):
        # Equal logits: argmax is arbitrary. Under shift, the class with
        # highest p_split / p_train ratio must win.
        logits = torch.zeros(1, 4)
        p_split = torch.tensor([0.1, 0.1, 0.7, 0.1])
        shifted = apply_multiclass_log_prior_shift(logits, p_split)
        assert shifted.argmax(dim=-1).item() == 2

    def test_shape_mismatch_raises(self):
        logits = torch.zeros(2, 4)
        with pytest.raises(ValueError):
            apply_multiclass_log_prior_shift(logits, torch.tensor([0.5, 0.5]))

    def test_dtype_and_device_preserved(self):
        logits = torch.zeros(2, 4, dtype=torch.float32)
        p_split = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
        shifted = apply_multiclass_log_prior_shift(logits, p_split)
        assert shifted.dtype == logits.dtype
        assert shifted.device == logits.device


class TestEndToEnd:
    def test_uniform_everything_preserves_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 4)
        p_split = uniform_multiclass_prior(4)
        shifted = apply_multiclass_log_prior_shift(logits, p_split)
        assert torch.equal(logits.argmax(dim=-1), shifted.argmax(dim=-1))

    def test_split_matching_train_preserves_argmax(self):
        # If split prior == train prior, nothing changes.
        torch.manual_seed(0)
        logits = torch.randn(32, 4)
        p_train = torch.tensor([0.4, 0.3, 0.2, 0.1])
        shifted = apply_multiclass_log_prior_shift(logits, p_split=p_train, p_train=p_train)
        assert torch.allclose(shifted, logits, atol=1e-6)

    def test_recalibration_can_flip_argmax(self):
        # Construct logits that argmax to class 0 pre-shift but class 3 post-shift
        # when p_split heavily favors class 3.
        logits = torch.tensor([[1.0, 0.5, 0.5, 0.0]])
        assert logits.argmax(dim=-1).item() == 0
        p_split = torch.tensor([0.01, 0.01, 0.01, 0.97])
        shifted = apply_multiclass_log_prior_shift(logits, p_split)
        assert shifted.argmax(dim=-1).item() == 3
