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

class MRFWrapper:
    """Wrapper to make tsai's PyTorch MiniRocket look identical to sktime for eval_rocket.py"""
    
    # 1. Update the signature to accept the fitted mrf and device
    def __init__(self, mrf, device):
        self.device = device
        self.mrf = mrf
        
        # We completely removed the redundant Ridge pipeline that was crashing!
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
            
        # Ensure the trained MRF module is on the correct device
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
        # Define a logarithmic search space for the regularization strength
        alpha_space = np.logspace(-3, 3, 7)
        self.classifier = make_pipeline(
            VarianceThreshold(threshold=1e-5),
            RidgeClassifierCV(alphas=alpha_space, class_weight="balanced")
        )
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
        # Assumes get_minirocket_features is imported from tsai
        X_feat = get_minirocket_features(X_train, self.mrf, chunksize=10, to_np=True)
        
        # FIX: Squeeze out the trailing dimension from tsai's output
        if X_feat.ndim == 3:
            X_feat = X_feat.squeeze(-1)
        
        t1 = time.time()
        print(f"  -> [Diagnostics] GPU Transformation complete in {t1 - t0:.2f} seconds.", flush=True)
        print(f"  -> [Diagnostics] New Feature Matrix Shape: {X_feat.shape}", flush=True)
        print("  -> [Diagnostics] Starting scikit-learn RidgeCV Fitting...", flush=True)
        
        # This single call now fits the VarianceThreshold AND the RidgeCV
        self.classifier.fit(X_feat, y_train)
        
        # Extract the optimal alpha chosen by RidgeCV for your logs
        best_alpha = self.classifier.named_steps['ridgeclassifiercv'].alpha_
        
        t2 = time.time()
        print(f"  -> [Diagnostics] RidgeCV fit complete in {t2 - t1:.2f} seconds.", flush=True)
        print(f"  -> [Diagnostics] Optimal alpha selected: {best_alpha}", flush=True)
        
        self.is_fitted = True
        
        # Free up GPU memory
        self.mrf = self.mrf.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
