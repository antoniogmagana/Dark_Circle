import torch
import torch.nn as nn
import torch.nn.functional as F

from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV

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


class ClassificationMiniRocket:
    """MiniRocket + RidgeClassifierCV"""
    def __init__(self, in_channels=None, num_classes=None, use_mel=False):
        self.transformer = MiniRocket(num_kernels=config.ROCKET_NUM_KERNELS)
        # FORCE 5-folds to prevent the SVD memory explosion
        self.classifier = RidgeClassifierCV(alphas=config.ROCKET_ALPHAS, cv=config.ROCKET_CV_FOLDS)
        self.is_fitted = False

    def fit(self, X_train, y_train):
        X_feat = self.transformer.fit_transform(X_train)
        self.classifier.fit(X_feat, y_train)
        self.is_fitted = True

    def predict(self, X_test):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X_feat = self.transformer.transform(X_test)
        return self.classifier.predict(X_feat)


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
