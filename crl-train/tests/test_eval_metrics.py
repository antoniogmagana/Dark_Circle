"""Tests for eval.py metric functions that don't require loading a model."""
import pytest
import torch

from eval import binary_metrics, multiclass_metrics, recalibrated_multiclass_metrics


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
