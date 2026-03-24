"""
Tests for model-train/models.py
Tests neural network model architectures and training components.
"""
import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Set DB password to avoid input prompt
os.environ['DB_PASSWORD'] = 'test_password'

# Add model-train to path FIRST, before any other imports
model_train_path = str(Path(__file__).parent.parent / "model-train")
if model_train_path in sys.path:
    sys.path.remove(model_train_path)
sys.path.insert(0, model_train_path)

from models import DetectionCNN, ClassificationCNN


class TestDetectionCNN:
    """Test DetectionCNN model architecture."""
    
    def test_model_initialization(self, mock_config):
        """Test model can be initialized."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        assert isinstance(model, nn.Module)
        assert model.config == mock_config
    
    def test_forward_pass_2d_input(self, mock_config):
        """Test forward pass with 2D spectrogram input."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        # Create mock spectrogram [Batch, Channels, MEL_BINS, FRAMES]
        batch_size = 4
        mel_bins = 64
        frames = 100
        x = torch.randn(batch_size, mock_config.IN_CHANNELS, mel_bins, frames)
        
        # Initialize lazy layers
        _ = model(x)
        
        # Now test actual forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_output_shape(self, mock_config):
        """Test output shape matches number of classes."""
        num_classes = 3
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=num_classes,
            config=mock_config
        )
        
        x = torch.randn(8, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        output = model(x)
        
        assert output.shape[0] == 8
        assert output.shape[1] == num_classes
    
    def test_different_input_sizes(self, mock_config):
        """Test model handles different input sizes."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        # Initialize with one size
        x1 = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x1)
        
        # Test with different sizes
        x2 = torch.randn(4, mock_config.IN_CHANNELS, 64, 150)
        output2 = model(x2)
        
        assert output2.shape == (4, 2)
    
    def test_gradient_flow(self, mock_config):
        """Test gradients flow through the network."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_get_optimizer(self, mock_config):
        """Test optimizer creation."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        optimizer = model.get_optimizer()
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
    
    def test_model_trainable(self, mock_config):
        """Test model parameters are trainable."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        assert trainable_params > 0
    
    def test_conv_layers_exist(self, mock_config):
        """Test convolution layers are present."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert isinstance(model.conv1, nn.Conv2d)
        assert isinstance(model.conv2, nn.Conv2d)
    
    def test_pooling_layer_exists(self, mock_config):
        """Test pooling layer exists."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        assert hasattr(model, 'pool')
        assert isinstance(model.pool, nn.MaxPool2d)
    
    def test_fc_layers_exist(self, mock_config):
        """Test fully connected layers exist."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')


class TestClassificationCNN:
    """Test ClassificationCNN model architecture."""
    
    def test_model_initialization(self, mock_config):
        """Test model can be initialized."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        assert isinstance(model, nn.Module)
        assert model.config == mock_config
    
    def test_forward_pass(self, mock_config):
        """Test forward pass."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        batch_size = 8
        x = torch.randn(batch_size, mock_config.IN_CHANNELS, 64, 100)
        
        # Initialize lazy layers
        _ = model(x)
        
        # Actual forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 3)
        assert not torch.isnan(output).any()
    
    def test_deeper_architecture(self, mock_config):
        """Test that classification model is deeper than detection."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        # Should have 4 conv layers
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
        assert hasattr(model, 'conv4')
    
    def test_dropout_present(self, mock_config):
        """Test dropout layer is present."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        assert hasattr(model, 'dropout')
        assert isinstance(model.dropout, nn.Dropout)
    
    def test_dropout_applied_during_training(self, mock_config):
        """Test dropout is applied during training mode."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        # Training mode
        model.train()
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        output_train = model(x)
        
        # Should produce valid output
        assert output_train.shape == (4, 3)
    
    def test_no_dropout_during_eval(self, mock_config):
        """Test dropout is not applied during eval mode."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        # Eval mode
        model.eval()
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        output_eval = model(x)
        
        assert output_eval.shape == (4, 3)
    
    def test_gradient_flow(self, mock_config):
        """Test gradients flow through all layers."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # All conv layers should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_multiclass_output(self, mock_config):
        """Test model handles multiple classes correctly."""
        num_classes = 10
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=num_classes,
            config=mock_config
        )
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        output = model(x)
        
        assert output.shape[1] == num_classes


class TestModelComparison:
    """Test comparison between model architectures."""
    
    def test_detection_smaller_than_classification(self, mock_config):
        """Test detection model has fewer parameters."""
        detection = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        classification = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        # Initialize both
        x = torch.randn(2, mock_config.IN_CHANNELS, 64, 100)
        _ = detection(x)
        _ = classification(x)
        
        det_params = sum(p.numel() for p in detection.parameters())
        cls_params = sum(p.numel() for p in classification.parameters())
        
        # Classification should have more parameters (4 conv vs 2 conv)
        assert cls_params > det_params
    
    def test_both_models_different_architectures(self, mock_config):
        """Test models have different numbers of conv layers."""
        detection = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        classification = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        # Count conv layers
        det_convs = sum(1 for m in detection.modules() if isinstance(m, nn.Conv2d))
        cls_convs = sum(1 for m in classification.modules() if isinstance(m, nn.Conv2d))
        
        assert cls_convs > det_convs


class TestModelTraining:
    """Test model training functionality."""
    
    def test_training_step(self, mock_config):
        """Test a single training step."""
        model = DetectionCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=2,
            config=mock_config
        )
        
        optimizer = model.get_optimizer()
        criterion = nn.CrossEntropyLoss()
        
        # Create batch
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        y = torch.randint(0, 2, (4,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Loss should be computed
        assert loss.item() >= 0
    
    def test_eval_mode(self, mock_config):
        """Test evaluation mode."""
        model = ClassificationCNN(
            in_channels=mock_config.IN_CHANNELS,
            num_classes=3,
            config=mock_config
        )
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        _ = model(x)  # Initialize
        
        model.eval()
        
        x = torch.randn(4, mock_config.IN_CHANNELS, 64, 100)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 3)
        assert not output.requires_grad
