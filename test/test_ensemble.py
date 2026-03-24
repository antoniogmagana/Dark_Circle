"""
Tests for ensemble-train/ensemble.py and ensemble prediction functionality.
Tests late fusion, model discovery, and two-stage prediction.
"""
import pytest
import torch
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add ensemble-train to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ensemble-train"))

from ensemble import (
    discover_models,
    parse_eval_results,
    weighted_late_fusion,
    two_stage_predict
)


class TestDiscoverModels:
    """Test model discovery in evaluation directory."""
    
    def test_discover_detection_models(self):
        """Test discovering detection models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake model files
            model_dir = Path(tmpdir) / "evaluations" / "detection"
            model_dir.mkdir(parents=True)
            
            (model_dir / "detection_model1.pth").touch()
            (model_dir / "detection_model2.pth").touch()
            (model_dir / "not_a_model.txt").touch()
            
            models = discover_models(tmpdir, "detection")
            
            assert len(models) == 2
            assert all(m.name.endswith('.pth') for m in models)
    
    def test_discover_category_models(self):
        """Test discovering category models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "evaluations" / "category"
            model_dir.mkdir(parents=True)
            
            (model_dir / "category_model1.pth").touch()
            
            models = discover_models(tmpdir, "category")
            
            assert len(models) == 1
    
    def test_discover_no_models(self):
        """Test handling when no models found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = discover_models(tmpdir, "detection")
            
            assert len(models) == 0
    
    def test_discover_instance_models(self):
        """Test discovering instance models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "evaluations" / "instance"
            model_dir.mkdir(parents=True)
            
            (model_dir / "instance_mustang.pth").touch()
            (model_dir / "instance_bicycle.pth").touch()
            
            models = discover_models(tmpdir, "instance")
            
            assert len(models) == 2


class TestParseEvalResults:
    """Test parsing evaluation results files."""
    
    def test_parse_valid_results(self):
        """Test parsing valid evaluation results."""
        with tempfile.TemporaryFile(mode='w+') as f:
            f.write("Model: detection_model1\n")
            f.write("Accuracy: 0.95\n")
            f.write("Precision: 0.92\n")
            f.write("Recall: 0.88\n")
            f.write("F1: 0.90\n")
            f.seek(0)
            
            results = parse_eval_results(f.name)
            
            assert results['accuracy'] == 0.95
            assert results['precision'] == 0.92
            assert results['recall'] == 0.88
            assert results['f1'] == 0.90
    
    def test_parse_missing_metrics(self):
        """Test parsing when some metrics are missing."""
        with tempfile.TemporaryFile(mode='w+') as f:
            f.write("Model: test_model\n")
            f.write("Accuracy: 0.85\n")
            f.seek(0)
            
            results = parse_eval_results(f.name)
            
            assert results['accuracy'] == 0.85
            assert 'precision' not in results or results['precision'] == 0
    
    def test_parse_empty_file(self):
        """Test parsing empty file."""
        with tempfile.TemporaryFile(mode='w+') as f:
            f.seek(0)
            
            results = parse_eval_results(f.name)
            
            # Should return empty dict or default values
            assert isinstance(results, dict)
    
    def test_parse_malformed_results(self):
        """Test handling malformed results."""
        with tempfile.TemporaryFile(mode='w+') as f:
            f.write("Invalid line\n")
            f.write("Accuracy: not_a_number\n")
            f.seek(0)
            
            results = parse_eval_results(f.name)
            
            # Should handle gracefully
            assert isinstance(results, dict)


class TestWeightedLateFusion:
    """Test weighted late fusion of model predictions."""
    
    def test_simple_fusion(self):
        """Test fusion with equal weights."""
        # Create predictions [batch_size, num_classes]
        pred1 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])  # 2 samples, 2 classes
        pred2 = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        
        predictions = [pred1, pred2]
        weights = [0.5, 0.5]
        
        fused = weighted_late_fusion(predictions, weights)
        
        # Should be weighted average
        expected = 0.5 * pred1 + 0.5 * pred2
        assert torch.allclose(fused, expected, atol=1e-6)
    
    def test_fusion_different_weights(self):
        """Test fusion with different weights."""
        pred1 = torch.tensor([[1.0, 0.0]])
        pred2 = torch.tensor([[0.0, 1.0]])
        
        predictions = [pred1, pred2]
        weights = [0.7, 0.3]
        
        fused = weighted_late_fusion(predictions, weights)
        
        expected = 0.7 * pred1 + 0.3 * pred2
        assert torch.allclose(fused, expected, atol=1e-6)
    
    def test_fusion_three_models(self):
        """Test fusion with three models."""
        pred1 = torch.tensor([[0.6, 0.3, 0.1]])
        pred2 = torch.tensor([[0.2, 0.7, 0.1]])
        pred3 = torch.tensor([[0.3, 0.4, 0.3]])
        
        predictions = [pred1, pred2, pred3]
        weights = [0.5, 0.3, 0.2]
        
        fused = weighted_late_fusion(predictions, weights)
        
        assert fused.shape == pred1.shape
        # Weights should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6
    
    def test_fusion_batch(self):
        """Test fusion with batch of samples."""
        batch_size = 16
        num_classes = 5
        
        pred1 = torch.rand(batch_size, num_classes)
        pred2 = torch.rand(batch_size, num_classes)
        
        predictions = [pred1, pred2]
        weights = [0.6, 0.4]
        
        fused = weighted_late_fusion(predictions, weights)
        
        assert fused.shape == (batch_size, num_classes)
    
    def test_fusion_weights_sum_to_one(self):
        """Test that weights should sum to 1."""
        pred1 = torch.tensor([[0.5, 0.5]])
        pred2 = torch.tensor([[0.3, 0.7]])
        
        predictions = [pred1, pred2]
        weights = [0.8, 0.2]  # Sum = 1.0
        
        fused = weighted_late_fusion(predictions, weights)
        
        # Result should be valid
        assert fused.shape == pred1.shape
    
    def test_fusion_single_model(self):
        """Test fusion with single model (edge case)."""
        pred1 = torch.tensor([[0.8, 0.2]])
        
        predictions = [pred1]
        weights = [1.0]
        
        fused = weighted_late_fusion(predictions, weights)
        
        # Should return same prediction
        assert torch.allclose(fused, pred1, atol=1e-6)


class TestTwoStagePredict:
    """Test two-stage prediction (detection -> classification)."""
    
    @patch('ensemble.DetectionCNN')
    @patch('ensemble.ClassificationCNN')
    def test_two_stage_positive_detection(self, mock_clf, mock_det):
        """Test two-stage when vehicle is detected."""
        # Mock detection model
        detection_model = Mock()
        detection_model.return_value = torch.tensor([[0.2, 0.8]])  # Vehicle detected
        mock_det.return_value = detection_model
        
        # Mock classification model
        classification_model = Mock()
        classification_model.return_value = torch.tensor([[0.1, 0.7, 0.2]])  # Class 1
        mock_clf.return_value = classification_model
        
        # Input
        x = torch.randn(1, 3, 64, 100)
        
        # Two-stage prediction
        result = two_stage_predict(
            x,
            detection_model=detection_model,
            classification_model=classification_model
        )
        
        # Should return classification result
        assert result.shape == (1, 3)
        assert detection_model.called
        assert classification_model.called
    
    @patch('ensemble.DetectionCNN')
    def test_two_stage_negative_detection(self, mock_det):
        """Test two-stage when no vehicle detected."""
        # Mock detection model
        detection_model = Mock()
        detection_model.return_value = torch.tensor([[0.9, 0.1]])  # No vehicle
        mock_det.return_value = detection_model
        
        # Classification shouldn't be called
        classification_model = Mock()
        
        # Input
        x = torch.randn(1, 3, 64, 100)
        
        result = two_stage_predict(
            x,
            detection_model=detection_model,
            classification_model=classification_model
        )
        
        # Should return negative result without calling classifier
        assert result is not None
        assert detection_model.called
        # Classification should not be called
        assert not classification_model.called
    
    def test_two_stage_batch(self):
        """Test two-stage with batch input."""
        # Mock models
        detection_model = Mock()
        # Batch of 4: 2 detect vehicle, 2 don't
        detection_model.return_value = torch.tensor([
            [0.2, 0.8],  # Vehicle
            [0.9, 0.1],  # No vehicle
            [0.1, 0.9],  # Vehicle
            [0.7, 0.3]   # No vehicle
        ])
        
        classification_model = Mock()
        # Only called for detected vehicles
        classification_model.return_value = torch.tensor([
            [0.1, 0.7, 0.2],
            [0.3, 0.4, 0.3]
        ])
        
        # Input batch
        x = torch.randn(4, 3, 64, 100)
        
        result = two_stage_predict(
            x,
            detection_model=detection_model,
            classification_model=classification_model
        )
        
        # Should process all samples
        assert result is not None


class TestEnsembleIntegration:
    """Integration tests for ensemble system."""
    
    def test_full_ensemble_pipeline(self):
        """Test full ensemble prediction pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup directories
            eval_dir = Path(tmpdir) / "evaluations" / "detection"
            eval_dir.mkdir(parents=True)
            
            # Create fake models
            (eval_dir / "model1.pth").touch()
            (eval_dir / "model2.pth").touch()
            
            # Create fake eval results
            results_file = eval_dir / "model1_results.txt"
            with open(results_file, 'w') as f:
                f.write("Accuracy: 0.95\nF1: 0.92\n")
            
            # Discovery should work
            models = discover_models(tmpdir, "detection")
            assert len(models) == 2
            
            # Parse results should work
            results = parse_eval_results(str(results_file))
            assert 'accuracy' in results
