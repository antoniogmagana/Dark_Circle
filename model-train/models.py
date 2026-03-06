import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV

# =====================================================================
# 1. 2D CONVOLUTIONAL NETWORKS (For Mel-Spectrograms)
# =====================================================================

class DetectionCNN(nn.Module):
    """
    SHALLOW 2D CNN: Fast Binary Gate (Vehicle vs. No Vehicle).
    Expects input shape: (Batch, in_channels, Mel_Bins, Time_Steps)
    """
    def __init__(self, in_channels=1):
        super(DetectionCNN, self).__init__()
        # Shallow and wide for maximum inference speed
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # LazyLinear automatically calculates the flattened dimension
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 1) # Hardcoded to 1 for Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # Sigmoid squashes output to a probability between 0.0 and 1.0
        return torch.sigmoid(self.fc2(x))


class ClassificationCNN(nn.Module):
    """
    DEEP 2D CNN: Multi-class Vehicle Identifier.
    Expects input shape: (Batch, in_channels, Mel_Bins, Time_Steps)
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(ClassificationCNN, self).__init__()
        # Deeper architecture to extract complex phase differences between channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Returns raw logits for CrossEntropyLoss
        return self.fc2(x) 


# =====================================================================
# 2. 1D CONVOLUTIONAL NETWORKS (For Raw Waveforms/SMV)
# =====================================================================

class WaveformDetectionCNN(nn.Module):
    """
    SHALLOW 1D CNN: Runs directly on voltage arrays for ultra-fast detection.
    Expects input shape: (Batch, in_channels, Time_Steps)
    """
    def __init__(self, in_channels=1):
        super(WaveformDetectionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=64, stride=16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=4)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class WaveformClassificationCNN(nn.Module):
    """
    DEEP 1D CNN: Multi-class Identifier running directly on raw arrays.
    Expects input shape: (Batch, in_channels, Time_Steps)
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(WaveformClassificationCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=64, stride=8)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2)
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =====================================================================
# 3. LONG SHORT-TERM MEMORY NETWORKS (Temporal Sequences)
# =====================================================================

class DetectionLSTM(nn.Module):
    """
    LIGHTWEIGHT LSTM: Binary Gate looking for temporal wave patterns.
    Expects input shape: (Batch, Time_Steps, input_size) 
    """
    def __init__(self, input_size=1):
        super(DetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # Pass the final memory state through a Sigmoid
        return torch.sigmoid(self.fc(hn[-1]))


class ClassificationLSTM(nn.Module):
    """
    HEAVY LSTM: Multi-class Identifier analyzing complex rhythmic sequences.
    Expects input shape: (Batch, Time_Steps, input_size)
    """
    def __init__(self, input_size=1, num_classes=5):
        super(ClassificationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=128, num_layers=3, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = F.relu(self.fc1(hn[-1]))
        return self.fc2(x) 


# =====================================================================
# 4. miniROCKET (High-Speed CPU / Linear Kernels)
# =====================================================================

class DetectionMiniRocket:
    """
    BINARY miniROCKET: Ultra-fast CPU Gate using Ridge Classification.
    Expects input shape: (Batch, in_channels, Time_Steps)
    """
    def __init__(self):
        # sktime's MiniRocket naturally scales to handle multi-channel inputs
        self.transformer = MiniRocket()
        # Ridge Classifier automatically handles binary labels (0 or 1)
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
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


class ClassificationMiniRocket:
    """
    MULTI-CLASS miniROCKET: Extremely fast identifier without deep learning.
    Expects input shape: (Batch, in_channels, Time_Steps)
    """
    def __init__(self):
        self.transformer = MiniRocket()
        # RidgeClassifierCV is natively multi-class capable
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
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