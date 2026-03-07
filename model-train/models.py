import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV

# =====================================================================
# 1. GLOBAL LEARNING DEFAULTS
# =====================================================================
BASE_LR = 1e-3
BASE_DROPOUT = 0.3
NUM_EPOCHS = 5

# =====================================================================
# 2. ARCHITECTURAL KNOBS
# =====================================================================

# --- Detection CNN (2D Mel-Spectrogram) ---
DET_CNN_LR = BASE_LR
DET_CNN_CHANNELS = [16, 32]
DET_CNN_KERNELS = [5, 3]
DET_CNN_STRIDES = [2, 1]
DET_CNN_PADS = [2, 1]
DET_CNN_HIDDEN = 64

# --- Classification CNN (2D Mel-Spectrogram) ---
CLASS_CNN_LR = 5e-4
CLASS_CNN_CHANNELS = [32, 64, 128, 256]
CLASS_CNN_KERNEL = 3
CLASS_CNN_PAD = 1
CLASS_CNN_HIDDEN = 512
CLASS_CNN_DROPOUT = 0.4

# --- Waveform 1D CNN ---
WAVE_CNN_LR = BASE_LR
WAVE_CNN_CHANNELS = [32, 64, 128]
WAVE_CNN_KERNELS = [64, 32, 16]
WAVE_CNN_STRIDES = [8, 4, 2]
WAVE_CNN_HIDDEN = 256

# --- LSTM Networks ---
LSTM_LR = 1e-3
LSTM_CNN_CHANNELS = [16, 32]
LSTM_CNN_KERNELS = [32, 16]
LSTM_CNN_STRIDES = [8, 4]
LSTM_CNN_POOLS = [4, 2]
LSTM_HIDDEN = 128
LSTM_LAYERS = 3
LSTM_FC_DIM = 64
LSTM_DROPOUT = BASE_DROPOUT

# --- miniROCKET ---
ROCKET_NUM_KERNELS = 10000
ROCKET_ALPHAS = np.logspace(-3, 3, 10)


# =====================================================================
# 3. MEL-ONLY MODELS (Conv2d)
# =====================================================================


class DetectionCNN(nn.Module):
    """
    Mel-only detection CNN.
    Input: [B, C, MEL_BINS, FRAMES]
    """

    def __init__(self, in_channels, num_classes, use_mel=True):
        super().__init__()
        self.use_mel = True  # enforce mel-only

        self.conv1 = nn.Conv2d(
            in_channels,
            DET_CNN_CHANNELS[0],
            kernel_size=DET_CNN_KERNELS[0],
            stride=DET_CNN_STRIDES[0],
            padding=DET_CNN_PADS[0],
        )
        self.conv2 = nn.Conv2d(
            DET_CNN_CHANNELS[0],
            DET_CNN_CHANNELS[1],
            kernel_size=DET_CNN_KERNELS[1],
            stride=DET_CNN_STRIDES[1],
            padding=DET_CNN_PADS[1],
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(DET_CNN_HIDDEN)
        self.fc2 = nn.Linear(DET_CNN_HIDDEN, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ClassificationCNN(nn.Module):
    """
    Mel-only classification CNN.
    Input: [B, C, MEL_BINS, FRAMES]
    """

    def __init__(self, in_channels, num_classes, use_mel=True):
        super().__init__()
        self.use_mel = True  # enforce mel-only

        self.conv1 = nn.Conv2d(
            in_channels,
            CLASS_CNN_CHANNELS[0],
            kernel_size=CLASS_CNN_KERNEL,
            padding=CLASS_CNN_PAD,
        )
        self.conv2 = nn.Conv2d(
            CLASS_CNN_CHANNELS[0],
            CLASS_CNN_CHANNELS[1],
            kernel_size=CLASS_CNN_KERNEL,
            padding=CLASS_CNN_PAD,
        )
        self.conv3 = nn.Conv2d(
            CLASS_CNN_CHANNELS[1],
            CLASS_CNN_CHANNELS[2],
            kernel_size=CLASS_CNN_KERNEL,
            padding=CLASS_CNN_PAD,
        )
        self.conv4 = nn.Conv2d(
            CLASS_CNN_CHANNELS[2],
            CLASS_CNN_CHANNELS[3],
            kernel_size=CLASS_CNN_KERNEL,
            padding=CLASS_CNN_PAD,
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(CLASS_CNN_HIDDEN)
        self.fc2 = nn.Linear(CLASS_CNN_HIDDEN, num_classes)
        self.dropout = nn.Dropout(CLASS_CNN_DROPOUT)

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
# 4. WAVEFORM-ONLY MODELS (Conv1d)
# =====================================================================


class WaveformClassificationCNN(nn.Module):
    """
    Waveform-only 1D CNN.
    Input: [B, C, T]
    """

    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        self.use_mel = False  # enforce waveform-only

        self.conv1 = nn.Conv1d(
            in_channels,
            WAVE_CNN_CHANNELS[0],
            kernel_size=WAVE_CNN_KERNELS[0],
            stride=WAVE_CNN_STRIDES[0],
        )
        self.conv2 = nn.Conv1d(
            WAVE_CNN_CHANNELS[0],
            WAVE_CNN_CHANNELS[1],
            kernel_size=WAVE_CNN_KERNELS[1],
            stride=WAVE_CNN_STRIDES[1],
        )
        self.conv3 = nn.Conv1d(
            WAVE_CNN_CHANNELS[1],
            WAVE_CNN_CHANNELS[2],
            kernel_size=WAVE_CNN_KERNELS[2],
            stride=WAVE_CNN_STRIDES[2],
        )

        self.fc1 = nn.LazyLinear(WAVE_CNN_HIDDEN)
        self.fc2 = nn.Linear(WAVE_CNN_HIDDEN, num_classes)
        self.dropout = nn.Dropout(BASE_DROPOUT)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =====================================================================
# 5. LSTM MODELS (Waveform-only)
# =====================================================================


class ClassificationLSTM(nn.Module):
    """
    Waveform-only LSTM classifier.
    Input: [B, C, T]
    """

    def __init__(self, in_channels, num_classes, use_mel=False):
        super().__init__()
        self.use_mel = False  # enforce waveform-only

        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(
                in_channels,
                LSTM_CNN_CHANNELS[0],
                kernel_size=LSTM_CNN_KERNELS[0],
                stride=LSTM_CNN_STRIDES[0],
            ),
            nn.ReLU(),
            nn.MaxPool1d(LSTM_CNN_POOLS[0]),
            nn.Conv1d(
                LSTM_CNN_CHANNELS[0],
                LSTM_CNN_CHANNELS[1],
                kernel_size=LSTM_CNN_KERNELS[1],
                stride=LSTM_CNN_STRIDES[1],
            ),
            nn.ReLU(),
            nn.MaxPool1d(LSTM_CNN_POOLS[1]),
        )

        self.lstm = nn.LSTM(
            input_size=LSTM_CNN_CHANNELS[1],
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
        )
        self.fc1 = nn.Linear(LSTM_HIDDEN, LSTM_FC_DIM)
        self.fc2 = nn.Linear(LSTM_FC_DIM, num_classes)

    def forward(self, x):
        x = self.cnn_frontend(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        x = F.relu(self.fc1(hn[-1]))
        return self.fc2(x)


# =====================================================================
# 6. miniROCKET (Non-PyTorch)
# =====================================================================


class ClassificationMiniRocket:
    """
    Non-neural model: MiniRocket + RidgeClassifierCV.
    """

    def __init__(self):
        self.transformer = MiniRocket(num_kernels=ROCKET_NUM_KERNELS)
        self.classifier = RidgeClassifierCV(alphas=ROCKET_ALPHAS)
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
# 7. MODEL REGISTRY
# =====================================================================

MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "ClassificationMiniRocket": ClassificationMiniRocket,
    # ADD MORE AS NEEDED
}
