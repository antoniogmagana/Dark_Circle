import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures


# =====================================================================
# Helpers
# =====================================================================

def _init_weights(module):
    """Kaiming initialization for Conv layers, Xavier for Linear layers."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =====================================================================
# 1. 2D MODELS (Mel-Spectrogram Input)
# =====================================================================

class DetectionCNN(nn.Module):
    """Binary detection CNN on mel spectrograms.  [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, config, use_mel=True):
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, config.CHANNELS[0],
                      kernel_size=config.KERNELS[0],
                      stride=config.STRIDES[0],
                      padding=config.PADS[0]),
            nn.BatchNorm2d(config.CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(config.CHANNELS[0], config.CHANNELS[1],
                      kernel_size=config.KERNELS[1],
                      stride=config.STRIDES[1],
                      padding=config.PADS[1]),
            nn.BatchNorm2d(config.CHANNELS[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(config.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(config.HIDDEN, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class ClassificationCNN(nn.Module):
    """Multi-class CNN on mel spectrograms.  [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, config, use_mel=True):
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, config.CHANNELS[0],
                      kernel_size=config.KERNEL, padding=config.PADS),
            nn.BatchNorm2d(config.CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(config.CHANNELS[0], config.CHANNELS[1],
                      kernel_size=config.KERNEL, padding=config.PADS),
            nn.BatchNorm2d(config.CHANNELS[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(config.CHANNELS[1], config.CHANNELS[2],
                      kernel_size=config.KERNEL, padding=config.PADS),
            nn.BatchNorm2d(config.CHANNELS[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(config.CHANNELS[2], config.CHANNELS[3],
                      kernel_size=config.KERNEL, padding=config.PADS),
            nn.BatchNorm2d(config.CHANNELS[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(config.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 2. 1D MODELS (Raw Waveform Input)
# =====================================================================

class WaveformClassificationCNN(nn.Module):
    """Multi-class CNN on raw waveforms.  [B, C, T]"""

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, config.CHANNELS[0],
                      kernel_size=config.KERNELS[0], stride=config.STRIDES[0]),
            nn.BatchNorm1d(config.CHANNELS[0]),
            nn.ReLU(inplace=True),

            nn.Conv1d(config.CHANNELS[0], config.CHANNELS[1],
                      kernel_size=config.KERNELS[1], stride=config.STRIDES[1]),
            nn.BatchNorm1d(config.CHANNELS[1]),
            nn.ReLU(inplace=True),

            nn.Conv1d(config.CHANNELS[1], config.CHANNELS[2],
                      kernel_size=config.KERNELS[2], stride=config.STRIDES[2]),
            nn.BatchNorm1d(config.CHANNELS[2]),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(config.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class ClassificationLSTM(nn.Module):
    """CNN front-end → LSTM on raw waveforms.  [B, C, T]"""

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config

        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(in_channels, config.CHANNELS[0],
                      kernel_size=config.KERNELS[0], stride=config.STRIDES[0]),
            nn.BatchNorm1d(config.CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(config.POOLS[0]),

            nn.Conv1d(config.CHANNELS[0], config.CHANNELS[1],
                      kernel_size=config.KERNELS[1], stride=config.STRIDES[1]),
            nn.BatchNorm1d(config.CHANNELS[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(config.POOLS[1]),
        )

        self.lstm = nn.LSTM(
            input_size=config.CHANNELS[1],
            hidden_size=config.HIDDEN,
            num_layers=config.LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.LAYERS > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN, config.DIM),
            nn.ReLU(inplace=True),
            nn.Linear(config.DIM, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.cnn_frontend(x)
        x = x.transpose(1, 2)                       # [B, T', C]
        _, (hn, _) = self.lstm(x)                    # hn: [layers, B, H]
        return self.head(hn[-1])

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 3. NON-PYTORCH MODELS
# =====================================================================

class IterativeMiniRocket(nn.Module):
    """
    End-to-End MiniRocket: frozen random-kernel features → trainable linear head.
    Expects 1D Waveform: [B, C, T]
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config
        self.c_in = in_channels
        self.seq_len = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
        self.num_features = getattr(config, "MINIROCKET_FEATURES", 10_000)

        self.mrf = MiniRocketFeatures(
            c_in=self.c_in, seq_len=self.seq_len, num_features=self.num_features
        )
        self.fc = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(getattr(config, "DROPOUT", 0.3))
        self.is_fitted = False

    def fit_extractor(self, dummy_batch):
        """Calculate random dilations and biases from a small data sample."""
        if self.is_fitted:
            return
        print("  -> [MiniRocket] Initializing random convolution kernels...", flush=True)
        self.mrf.fit(dummy_batch)
        self.is_fitted = True
        for param in self.mrf.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.mrf(x)
            if features.ndim == 3:
                features = features.squeeze(-1)
        return self.fc(self.dropout(features))

    def get_optimizer(self):
        return optim.Adam(self.fc.parameters(), lr=1e-3, weight_decay=1e-3)


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


def build_model(input_channels, num_classes, config):
    model_name = config.MODEL_NAME
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](
        in_channels=input_channels,
        num_classes=num_classes,
        config=config,
        use_mel=getattr(config, "USE_MEL", True),
    )
