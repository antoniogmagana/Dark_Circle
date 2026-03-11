import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold

import config

# =====================================================================
# 1. 2D MODELS (Mel-Spectrogram Input)
# =====================================================================


class DetectionCNN(nn.Module):
    """Expects 2D Spectrogram: [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, use_mel=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            config.CHANNELS[0],
            kernel_size=config.KERNELS[0],
            stride=config.STRIDES[0],
            padding=config.PADS[0],
        )
        self.conv2 = nn.Conv2d(
            config.CHANNELS[0],
            config.CHANNELS[1],
            kernel_size=config.KERNELS[1],
            stride=config.STRIDES[1],
            padding=config.PADS[1],
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(config.HIDDEN)
        self.fc2 = nn.Linear(config.HIDDEN, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def get_optimizer(self):
        self.optimizer = optim.Adam
        return self.optimizer(self.parameters(), lr=config.LEARNING_RATE)


class ClassificationCNN(nn.Module):
    """Expects 2D Spectrogram: [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, use_mel=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            config.CHANNELS[0],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.conv2 = nn.Conv2d(
            config.CHANNELS[0],
            config.CHANNELS[1],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.conv3 = nn.Conv2d(
            config.CHANNELS[1],
            config.CHANNELS[2],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.conv4 = nn.Conv2d(
            config.CHANNELS[2],
            config.CHANNELS[3],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(config.HIDDEN)
        self.fc2 = nn.Linear(config.HIDDEN, num_classes)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_optimizer(self):
        self.optimizer = optim.Adam
        return self.optimizer(self.parameters(), lr=config.LEARNING_RATE)

# =====================================================================
# 2. 1D MODELS (Raw Waveform Input)
# =====================================================================


class WaveformClassificationCNN(nn.Module):
    """Expects 1D Waveform: [B, C, T]"""

    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            config.CHANNELS[0],
            kernel_size=config.KERNELS[0],
            stride=config.STRIDES[0],
        )
        self.conv2 = nn.Conv1d(
            config.CHANNELS[0],
            config.CHANNELS[1],
            kernel_size=config.KERNELS[1],
            stride=config.STRIDES[1],
        )
        self.conv3 = nn.Conv1d(
            config.CHANNELS[1],
            config.CHANNELS[2],
            kernel_size=config.KERNELS[2],
            stride=config.STRIDES[2],
        )

        self.fc1 = nn.LazyLinear(config.HIDDEN)
        self.fc2 = nn.Linear(config.HIDDEN, num_classes)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_optimizer(self):
        self.optimizer = optim.Adam
        return self.optimizer(self.parameters(), lr=config.LEARNING_RATE)

class ClassificationLSTM(nn.Module):
    """Expects 1D Waveform: [B, C, T]"""

    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        # self.optimizer = self._get_optimizer()
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(
                in_channels,
                config.CHANNELS[0],
                kernel_size=config.KERNELS[0],
                stride=config.STRIDES[0],
            ),
            nn.ReLU(),
            nn.MaxPool1d(config.POOLS[0]),
            nn.Conv1d(
                config.CHANNELS[0],
                config.CHANNELS[1],
                kernel_size=config.KERNELS[1],
                stride=config.STRIDES[1],
            ),
            nn.ReLU(),
            nn.MaxPool1d(config.POOLS[1]),
        )

        self.lstm = nn.LSTM(
            input_size=config.CHANNELS[1],
            hidden_size=config.HIDDEN,
            num_layers=config.LAYERS,
            batch_first=True,
            dropout=config.DROPOUT,
        )
        self.fc1 = nn.Linear(config.HIDDEN, config.DIM)
        self.fc2 = nn.Linear(config.DIM, num_classes)

    def forward(self, x):
        x = self.cnn_frontend(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        x = F.relu(self.fc1(hn[-1]))
        return self.fc2(x)

    def get_optimizer(self):
        self.optimizer = optim.Adam
        return self.optimizer(self.parameters(), lr=config.LEARNING_RATE)

# =====================================================================
# 3. NON-PYTORCH MODELS
# =====================================================================

# Iterative adaptation of Mini Rocket wrapped in pytorch model architecture courtesy of tsai
class IterativeMiniRocket(nn.Module):
    """
    End-to-End PyTorch MiniRocket.
    Extracts features batch-by-batch on the GPU and trains a linear head iteratively.
    Expects 1D Waveform: [B, C, T]
    """
    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        self.c_in = in_channels
        self.seq_len = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
        
        # 1. The feature extractor (will be frozen)
        self.mrf = MiniRocketFeatures(c_in=self.c_in, seq_len=self.seq_len)
        
        # 2. The Trainable Classification Head
        self.fc = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(config.DROPOUT if hasattr(config, 'DROPOUT') else 0.3)
        
        self.is_fitted = False

    def fit_extractor(self, dummy_batch):
        """Calculates random dilations and biases from a small data sample."""
        if not self.is_fitted:
            print("  -> [MiniRocket] Initializing random convolution kernels...", flush=True)
            self.mrf.fit(dummy_batch)
            self.is_fitted = True
            
            # Freeze the MRF convolutions so we ONLY train the Linear head
            for param in self.mrf.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 1. Extract MiniRocket features (no gradients needed here)
        with torch.no_grad():
            features = self.mrf(x)
            # tsai MRF outputs [B, Features, 1], we need to flatten the trailing dimension
            if features.ndim == 3:
                features = features.squeeze(-1)
                
        # 2. Pass through the trainable head
        x = self.dropout(features)
        return self.fc(x)

    def get_optimizer(self):
        # Crucial: We only pass the linear layer parameters to the optimizer!
        self.optimizer = optim.Adam
        return self.optimizer(self.fc.parameters(), lr=1e-3, weight_decay=1e-3)


# =====================================================================
# 4. MODEL REGISTRY
# =====================================================================

MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "IterativeMiniRocket": IterativeMiniRocket,
}


def build_model(input_channels, num_classes):
    model_name = config.MODEL_NAME

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    ModelClass = MODEL_REGISTRY[model_name]

    return ModelClass(
        in_channels=input_channels, num_classes=num_classes, use_mel=config.USE_MEL
    )
