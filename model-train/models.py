import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
from sklearn.linear_model import RidgeClassifier

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
            config.DET_CNN_CHANNELS[0],
            kernel_size=config.DET_CNN_KERNELS[0],
            stride=config.DET_CNN_STRIDES[0],
            padding=config.DET_CNN_PADS[0],
        )
        self.conv2 = nn.Conv2d(
            config.DET_CNN_CHANNELS[0],
            config.DET_CNN_CHANNELS[1],
            kernel_size=config.DET_CNN_KERNELS[1],
            stride=config.DET_CNN_STRIDES[1],
            padding=config.DET_CNN_PADS[1],
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(config.DET_CNN_HIDDEN)
        self.fc2 = nn.Linear(config.DET_CNN_HIDDEN, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ClassificationCNN(nn.Module):
    """Expects 2D Spectrogram: [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, use_mel=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            config.CLASS_CNN_CHANNELS[0],
            kernel_size=config.CLASS_CNN_KERNEL,
            padding=config.CLASS_CNN_PAD,
        )
        self.conv2 = nn.Conv2d(
            config.CLASS_CNN_CHANNELS[0],
            config.CLASS_CNN_CHANNELS[1],
            kernel_size=config.CLASS_CNN_KERNEL,
            padding=config.CLASS_CNN_PAD,
        )
        self.conv3 = nn.Conv2d(
            config.CLASS_CNN_CHANNELS[1],
            config.CLASS_CNN_CHANNELS[2],
            kernel_size=config.CLASS_CNN_KERNEL,
            padding=config.CLASS_CNN_PAD,
        )
        self.conv4 = nn.Conv2d(
            config.CLASS_CNN_CHANNELS[2],
            config.CLASS_CNN_CHANNELS[3],
            kernel_size=config.CLASS_CNN_KERNEL,
            padding=config.CLASS_CNN_PAD,
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(config.CLASS_CNN_HIDDEN)
        self.fc2 = nn.Linear(config.CLASS_CNN_HIDDEN, num_classes)
        self.dropout = nn.Dropout(config.CLASS_CNN_DROPOUT)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =====================================================================
# 2. 1D MODELS (Raw Waveform Input)
# =====================================================================


class WaveformClassificationCNN(nn.Module):
    """Expects 1D Waveform: [B, C, T]"""

    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            config.WAVE_CNN_CHANNELS[0],
            kernel_size=config.WAVE_CNN_KERNELS[0],
            stride=config.WAVE_CNN_STRIDES[0],
        )
        self.conv2 = nn.Conv1d(
            config.WAVE_CNN_CHANNELS[0],
            config.WAVE_CNN_CHANNELS[1],
            kernel_size=config.WAVE_CNN_KERNELS[1],
            stride=config.WAVE_CNN_STRIDES[1],
        )
        self.conv3 = nn.Conv1d(
            config.WAVE_CNN_CHANNELS[1],
            config.WAVE_CNN_CHANNELS[2],
            kernel_size=config.WAVE_CNN_KERNELS[2],
            stride=config.WAVE_CNN_STRIDES[2],
        )

        self.fc1 = nn.LazyLinear(config.WAVE_CNN_HIDDEN)
        self.fc2 = nn.Linear(config.WAVE_CNN_HIDDEN, num_classes)
        self.dropout = nn.Dropout(config.BASE_DROPOUT)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ClassificationLSTM(nn.Module):
    """Expects 1D Waveform: [B, C, T]"""

    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(
                in_channels,
                config.LSTM_CNN_CHANNELS[0],
                kernel_size=config.LSTM_CNN_KERNELS[0],
                stride=config.LSTM_CNN_STRIDES[0],
            ),
            nn.ReLU(),
            nn.MaxPool1d(config.LSTM_CNN_POOLS[0]),
            nn.Conv1d(
                config.LSTM_CNN_CHANNELS[0],
                config.LSTM_CNN_CHANNELS[1],
                kernel_size=config.LSTM_CNN_KERNELS[1],
                stride=config.LSTM_CNN_STRIDES[1],
            ),
            nn.ReLU(),
            nn.MaxPool1d(config.LSTM_CNN_POOLS[1]),
        )

        self.lstm = nn.LSTM(
            input_size=config.LSTM_CNN_CHANNELS[1],
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT,
        )
        self.fc1 = nn.Linear(config.LSTM_HIDDEN, config.LSTM_FC_DIM)
        self.fc2 = nn.Linear(config.LSTM_FC_DIM, num_classes)

    def forward(self, x):
        x = self.cnn_frontend(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        x = F.relu(self.fc1(hn[-1]))
        return self.fc2(x)


# =====================================================================
# 3. NON-PYTORCH MODELS
# =====================================================================

class MRFWrapper:
    """Wrapper to make tsai's PyTorch MiniRocket look identical to sktime for eval_rocket.py"""
    def __init__(self, mrf, device):
        self.mrf = mrf
        self.device = device
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
            
        self.mrf = self.mrf.to(self.device)
        X_feat = get_minirocket_features(X, self.mrf, chunksize=10, to_np=True)
        
        # FIX: Squeeze out the trailing dimension from tsai's output
        if X_feat.ndim == 3:
            X_feat = X_feat.squeeze(-1)
            
        return X_feat

class ClassificationMiniRocket:
    """GPU-Accelerated MiniRocket + RidgeClassifier"""
    def __init__(self, in_channels=None, num_classes=None, use_mel=False):
        self.c_in = config.IN_CHANNELS
        self.seq_len = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
        self.device = config.DEVICE
        
        self.mrf = MiniRocketFeatures(c_in=self.c_in, seq_len=self.seq_len).to(self.device)
        self.classifier = RidgeClassifier(alpha=1.0) 
        self.is_fitted = False

    def fit(self, X_train, y_train):
        print("\n  -> [Diagnostics] Starting tsai GPU Feature Transformation...", flush=True)
        t0 = time.time()
        
        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        else:
            X_train = X_train.to(self.device)
            
        print("  -> [Diagnostics] Fitting MRF biases...", flush=True)
        self.mrf.fit(X_train[:10])
        
        print("  -> [Diagnostics] Extracting features in safe chunks...", flush=True)
        X_feat = get_minirocket_features(X_train, self.mrf, chunksize=10, to_np=True)
        
        # FIX: Squeeze out the trailing dimension from tsai's output
        if X_feat.ndim == 3:
            X_feat = X_feat.squeeze(-1)
        
        t1 = time.time()
        print(f"  -> [Diagnostics] GPU Transformation complete in {t1 - t0:.2f} seconds.", flush=True)
        print(f"  -> [Diagnostics] New Feature Matrix Shape: {X_feat.shape}", flush=True)
        print("  -> [Diagnostics] Starting scikit-learn Ridge Fitting...", flush=True)
        
        self.classifier.fit(X_feat, y_train)
        
        t2 = time.time()
        print(f"  -> [Diagnostics] Ridge fit complete in {t2 - t1:.2f} seconds.", flush=True)
        self.is_fitted = True
        
        self.mrf = self.mrf.cpu()

    def predict(self, X_test):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        X_feat = self.transformer.transform(X_test)
        return self.classifier.predict(X_feat)
        
    @property
    def transformer(self):
        return MRFWrapper(self.mrf, self.device)


# =====================================================================
# 4. MODEL REGISTRY
# =====================================================================

MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "ClassificationMiniRocket": ClassificationMiniRocket,
}


def build_model(input_channels, num_classes):
    model_name = config.MODEL_NAME

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    ModelClass = MODEL_REGISTRY[model_name]

    if model_name == "ClassificationMiniRocket":
        return ModelClass()

    return ModelClass(
        in_channels=input_channels, num_classes=num_classes, use_mel=config.USE_MEL
    )
