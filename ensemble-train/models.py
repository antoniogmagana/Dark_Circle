import torch
import torch.nn as nn
import torch.optim as optim

from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures


# =====================================================================
# Helpers
# =====================================================================

def _init_weights(module):
    """Kaiming init for Conv, Xavier for Linear, ones/zeros for BatchNorm."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# =====================================================================
# 1. 2D MODEL — Mel-Spectrogram Input  [B, C, MEL_BINS, FRAMES]
# =====================================================================

class SpectrogramCNN(nn.Module):
    """
    Dynamic 2D CNN for mel spectrograms.

    The number of conv blocks is determined by len(config.CHANNELS).
    Uses AdaptiveAvgPool2d so it works on any spectrogram size — from
    a full 64×63 audio spectrogram to a compact 32×7 seismic one.

    Config fields:
        CHANNELS   list[int]  — output channels per conv block
        KERNELS    list[int]  — kernel size per block
        STRIDES    list[int]  — conv stride per block
        PADS       list[int]  — conv padding per block
        POOL       bool       — add MaxPool2d(2,2) after each block
                                (set False for small spectrograms)
        HIDDEN     int        — classifier hidden dim
        DROPOUT    float      — classifier dropout
    """

    def __init__(self, in_channels, num_classes, config, use_mel=True):
        super().__init__()
        self.config = config
        channels = config.CHANNELS
        use_pool = getattr(config, "POOL", True)

        layers = []
        prev_c = in_channels
        for i, c in enumerate(channels):
            layers.append(nn.Conv2d(
                prev_c, c,
                kernel_size=config.KERNELS[i],
                stride=config.STRIDES[i],
                padding=config.PADS[i],
            ))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            if use_pool:
                layers.append(nn.MaxPool2d(2, 2))
            prev_c = c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], config.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(getattr(config, "DROPOUT", 0.0)),
            nn.Linear(config.HIDDEN, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)    # [B, C_last]
        return self.classifier(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 2. 1D MODELS — Raw Waveform Input  [B, C, T]
# =====================================================================

class Waveform1DCNN(nn.Module):
    """
    Dynamic 1D CNN for raw waveforms.

    The number of conv blocks is determined by len(config.CHANNELS).
    Uses AdaptiveAvgPool1d so it works on any signal length — from
    200 samples (seismic) to 16,000 (audio).

    Config fields:
        CHANNELS   list[int]  — output channels per conv block
        KERNELS    list[int]  — kernel size per block
        STRIDES    list[int]  — conv stride per block
        HIDDEN     int        — classifier hidden dim
        DROPOUT    float      — classifier dropout
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config
        channels = config.CHANNELS

        layers = []
        prev_c = in_channels
        for i, c in enumerate(channels):
            layers.append(nn.Conv1d(
                prev_c, c,
                kernel_size=config.KERNELS[i],
                stride=config.STRIDES[i],
                padding=config.KERNELS[i] // 2,
            ))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            prev_c = c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], config.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)                # [B, C_last]
        return self.classifier(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class ClassificationLSTM(nn.Module):
    """
    CNN front-end → LSTM for sequential modelling of waveforms.

    The CNN downsamples the raw signal into a shorter sequence of
    feature vectors, which the LSTM processes as time steps.

    Config fields:
        CHANNELS   list[int]  — CNN output channels (2 layers)
        KERNELS    list[int]  — CNN kernel sizes
        STRIDES    list[int]  — CNN strides
        POOLS      list[int]  — MaxPool1d kernel per CNN layer
        HIDDEN     int        — LSTM hidden size
        LAYERS     int        — LSTM depth
        DIM        int        — classifier intermediate dim
        DROPOUT    float      — LSTM + classifier dropout
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config

        # Build CNN frontend dynamically from config lists
        cnn_layers = []
        prev_c = in_channels
        for i in range(len(config.CHANNELS)):
            cnn_layers.extend([
                nn.Conv1d(prev_c, config.CHANNELS[i],
                          kernel_size=config.KERNELS[i],
                          stride=config.STRIDES[i]),
                nn.BatchNorm1d(config.CHANNELS[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(config.POOLS[i]),
            ])
            prev_c = config.CHANNELS[i]

        self.cnn_frontend = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(
            input_size=config.CHANNELS[-1],
            hidden_size=config.HIDDEN,
            num_layers=config.LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.LAYERS > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN, config.DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
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
# 3. RESIDUAL 1D CNN — New
# =====================================================================

class _ResidualBlock1D(nn.Module):
    """
    Basic residual block: two conv layers with a skip connection.
    Handles channel expansion and stride-2 downsampling via a
    learned 1×1 projection on the skip path.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        # Skip connection: project if channels or length change
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))


class ResNet1D(nn.Module):
    """
    Lightweight residual 1D CNN.

    Particularly effective for short signals (seismic/accel at 200 samples)
    where skip connections preserve gradient flow through the limited
    temporal extent.  Also works well on long audio signals thanks to
    a larger stem stride.

    Architecture:
      Stem → [Stage_1 → Stage_2 → ...] → AdaptiveAvgPool → Head

    Each stage starts with a stride-2 residual block (downsamples + expands
    channels), followed by (BLOCKS_PER_STAGE - 1) same-resolution blocks.

    Config fields:
        CHANNELS          list[int]  — channel progression [stem, stage1, stage2, ...]
        STEM_KERNEL       int        — stem convolution kernel size
        STEM_STRIDE       int        — stem convolution stride
        BLOCKS_PER_STAGE  int        — residual blocks per stage (including downsample)
        HIDDEN            int        — classifier hidden dim
        DROPOUT           float      — classifier dropout
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config
        channels = config.CHANNELS
        blocks_per_stage = getattr(config, "BLOCKS_PER_STAGE", 2)

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0],
                      kernel_size=config.STEM_KERNEL,
                      stride=config.STEM_STRIDE,
                      padding=config.STEM_KERNEL // 2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        stages = []
        for i in range(1, len(channels)):
            # First block: stride-2 downsample + channel expansion
            stages.append(_ResidualBlock1D(
                channels[i - 1], channels[i], kernel_size=3, stride=2,
            ))
            # Remaining blocks: same resolution
            for _ in range(blocks_per_stage - 1):
                stages.append(_ResidualBlock1D(
                    channels[i], channels[i], kernel_size=3, stride=1,
                ))

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(channels[-1], config.HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)                # [B, C_last]
        return self.head(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 4. MINIROCKET
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
# 5. MODEL REGISTRY
# =====================================================================

# DetectionCNN and ClassificationCNN are both SpectrogramCNN —
# the config lookup table provides different hyperparameters for each name.
MODEL_REGISTRY = {
    "DetectionCNN":              SpectrogramCNN,
    "ClassificationCNN":         SpectrogramCNN,
    "WaveformClassificationCNN": Waveform1DCNN,
    "ClassificationLSTM":        ClassificationLSTM,
    "ResNet1D":                  ResNet1D,
    "IterativeMiniRocket":       IterativeMiniRocket,
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
