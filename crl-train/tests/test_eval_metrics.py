"""Tests for eval.py metric functions that don't require loading a model."""
import pytest
import torch

from eval import (
    binary_metrics, multiclass_metrics,
    recalibrated_binary_metrics, recalibrated_multiclass_metrics,
)


class TestMulticlassSupportOnly:
    def test_all_classes_present_equals_raw_macro(self):
        """When every class has support, macro_f1 == macro_f1_support_only."""
        logits = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],  # pred 0, label 0
            [0.0, 2.0, 0.0, 0.0],  # pred 1, label 1
            [0.0, 0.0, 2.0, 0.0],  # pred 2, label 2
            [0.0, 0.0, 0.0, 2.0],  # pred 3, label 3
        ])
        labels = torch.tensor([0, 1, 2, 3])
        m = multiclass_metrics(logits, labels, n_classes=4)
        assert m["macro_f1"] == m["macro_f1_support_only"]
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_missing_classes_inflate_support_only(self):
        """On a split with only 2 of 4 classes, support_only is higher than unfiltered."""
        logits = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ])
        labels = torch.tensor([0, 0, 2, 2])  # only classes 0 and 2 present
        m = multiclass_metrics(logits, labels, n_classes=4)
        assert m["macro_f1"] == pytest.approx(0.5)          # (1 + 0 + 1 + 0) / 4
        assert m["macro_f1_support_only"] == pytest.approx(1.0)  # (1 + 1) / 2

    def test_zero_support_overall_is_zero(self):
        """Edge case: no labeled samples."""
        logits = torch.empty(0, 4)
        labels = torch.empty(0, dtype=torch.long)
        m = multiclass_metrics(logits, labels, n_classes=4)
        assert m["macro_f1_support_only"] == 0.0


class TestBinaryDegeneracyDetection:
    """balanced_accuracy and MCC must expose degenerate classifiers that F1 hides."""

    def test_all_positive_predictor_f1_inflated_but_mcc_zero(self):
        # 70% positive class, predict positive for everything. F1 looks OK
        # (≈0.82), but the classifier is useless — bal_acc=0.5, mcc=0.
        logits = torch.full((100,), 10.0)  # all positive predictions
        labels = torch.cat([torch.ones(70), torch.zeros(30)]).long()
        m = binary_metrics(logits, labels)
        assert m["recall"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(0.0)
        assert m["f1"] > 0.8
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        assert m["mcc"] == pytest.approx(0.0)

    def test_all_negative_predictor_also_detected(self):
        logits = torch.full((100,), -10.0)
        labels = torch.cat([torch.ones(70), torch.zeros(30)]).long()
        m = binary_metrics(logits, labels)
        assert m["recall"] == pytest.approx(0.0)
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        assert m["mcc"] == pytest.approx(0.0)

    def test_perfect_classifier_all_metrics_max(self):
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        labels = torch.tensor([1, 1, 0, 0])
        m = binary_metrics(logits, labels)
        assert m["f1"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)

    def test_random_classifier_mcc_near_zero(self):
        torch.manual_seed(0)
        logits = torch.randn(1000)
        labels = torch.randint(0, 2, (1000,))
        m = binary_metrics(logits, labels)
        # Not strictly 0 due to finite sample, but small relative to 1.
        assert abs(m["mcc"]) < 0.15

    def test_inverted_classifier_has_negative_mcc(self):
        # Predict opposite of truth: mcc must be negative.
        logits = torch.tensor([-10.0, -10.0, 10.0, 10.0])
        labels = torch.tensor([1, 1, 0, 0])
        m = binary_metrics(logits, labels)
        assert m["mcc"] == pytest.approx(-1.0)


class TestRecalibratedBinaryRobustMetrics:
    def test_recalibrated_binary_emits_balanced_accuracy_and_mcc(self):
        logits = torch.randn(50)
        labels = torch.randint(0, 2, (50,))
        m = recalibrated_binary_metrics(logits, labels)
        assert "balanced_accuracy" in m
        assert "mcc" in m
        assert "specificity" in m


class TestRecalibratedInheritsSupportOnly:
    def test_recalibrated_metrics_include_support_only(self):
        """recalibrated_multiclass_metrics must surface the same support_only field."""
        torch.manual_seed(0)
        logits = torch.randn(16, 4)
        labels = torch.tensor([0, 1] * 8)  # only classes 0 and 1
        m = recalibrated_multiclass_metrics(logits, labels, n_classes=4)
        assert "macro_f1_support_only" in m
        assert "p_split" in m
        assert "p_train_assumed" in m
        # Labels only contain {0, 1}, so p_split has support on those
        assert m["p_split"][0] > 0.1
        assert m["p_split"][1] > 0.1
