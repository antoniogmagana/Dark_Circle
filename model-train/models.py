import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features

# NOTICE: global 'config' is no longer imported


# =====================================================================
# 1. 2D MODELS (Mel-Spectrogram Input)
# =====================================================================

class DetectionCNN(nn.Module):
    """Expects 2D Spectrogram: [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, config, use_mel=True):
        super().__init__()
        self.config = config
        
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
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class ClassificationCNN(nn.Module):
    """Expects 2D Spectrogram: [B, C, MEL_BINS, FRAMES]"""

    def __init__(self, in_channels, num_classes, config, use_mel=True):
        super().__init__()
        self.config = config
        
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
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 2. 1D MODELS (Raw Waveform Input)
# =====================================================================

class WaveformClassificationCNN(nn.Module):
    """Expects 1D Waveform: [B, C, T]"""

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config
        
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
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class ClassificationLSTM(nn.Module):
    """Expects 1D Waveform: [B, C, T]"""

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config
        
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
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 3. TIME SERIES MODELS (Raw Waveform Input)
# =====================================================================

class _InceptionBlock(nn.Module):
    """Single inception module: bottleneck → parallel multi-scale convs + maxpool branch."""

    def __init__(self, in_channels, nb_filters, kernels, bottleneck_size=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(bottleneck_size, nb_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        ])

        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm1d(nb_filters * (len(kernels) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        outs = [conv(bottleneck) for conv in self.conv_branches]
        outs.append(self.maxpool_branch(x))
        return self.act(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    """
    Multi-scale inception architecture for 1D time series.
    Parallel conv branches at different kernel sizes capture vehicle vibrations
    across multiple temporal scales. Expects 1D Waveform: [B, C, T]
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config

        nb_filters = config.NB_FILTERS
        kernels = config.INCEPTION_KERNELS
        n_blocks = config.INCEPTION_BLOCKS
        out_channels = nb_filters * (len(kernels) + 1)

        self.inception_blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleDict()

        ch = in_channels
        residual_ch = in_channels

        for i in range(n_blocks):
            self.inception_blocks.append(_InceptionBlock(ch, nb_filters, kernels))
            # Residual shortcut every 3 blocks
            if (i + 1) % 3 == 0:
                self.shortcuts[str(i)] = nn.Sequential(
                    nn.Conv1d(residual_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                residual_ch = out_channels
            ch = out_channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        residual = x
        for i, block in enumerate(self.inception_blocks):
            x = block(x)
            if str(i) in self.shortcuts:
                residual = self.shortcuts[str(i)](residual)
                x = F.relu(x + residual)
                residual = x

        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class _TemporalBlock(nn.Module):
    """Dilated causal residual block used by TCN."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        seq_len = x.size(-1)
        # Trim causal padding from the right to preserve sequence length
        out = self.act(self.conv1(x)[..., :seq_len])
        out = self.dropout(out)
        out = self.act(self.conv2(out)[..., :seq_len])
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with exponentially growing dilations.
    Captures long-range temporal patterns in vehicle seismic/acoustic signals
    while parallelizing like a CNN. Expects 1D Waveform: [B, C, T]
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config

        ch = config.TCN_CHANNELS
        ks = config.TCN_KERNEL_SIZE
        levels = config.TCN_LEVELS
        dropout = config.DROPOUT

        layers = []
        for i in range(levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else ch
            layers.append(_TemporalBlock(in_ch, ch, ks, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


class BiGRU(nn.Module):
    """
    CNN front-end + Bidirectional GRU.
    Identical CNN front-end to ClassificationLSTM for a clean A/B comparison.
    Bidirectional pass sees both onset and decay of vibration events.
    Expects 1D Waveform: [B, C, T]
    """

    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config

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

        self.gru = nn.GRU(
            input_size=config.CHANNELS[1],
            hidden_size=config.HIDDEN,
            num_layers=config.LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.LAYERS > 1 else 0.0,
        )

        self.fc1 = nn.Linear(config.HIDDEN * 2, config.DIM)
        self.fc2 = nn.Linear(config.DIM, num_classes)

    def forward(self, x):
        x = self.cnn_frontend(x)
        x = x.transpose(1, 2)          # [B, T', C]
        _, hn = self.gru(x)            # hn: [num_layers*2, B, HIDDEN]
        # Concatenate last forward and backward hidden states
        x = torch.cat([hn[-2], hn[-1]], dim=-1)  # [B, HIDDEN*2]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)


# =====================================================================
# 4. NON-PYTORCH MODELS
# =====================================================================

class IterativeMiniRocket(nn.Module):
    """
    End-to-End PyTorch MiniRocket.
    Extracts features batch-by-batch on the GPU and trains a linear head iteratively.
    Expects 1D Waveform: [B, C, T]
    """
    def __init__(self, in_channels, num_classes, config, use_mel=False):
        super().__init__()
        self.config = config
        self.c_in = in_channels
        self.seq_len = int(config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS)
        self.num_features = getattr(config, 'MINIROCKET_FEATURES', 10000)
        
        # 1. The feature extractor (will be frozen)
        self.mrf = MiniRocketFeatures(c_in=self.c_in, seq_len=self.seq_len, num_features=self.num_features)
        
        # 2. The Trainable Classification Head
        self.fc = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(getattr(config, 'DROPOUT', 0.3))
        
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
        return optim.Adam(self.fc.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-3)


# =====================================================================
# 4. MODEL REGISTRY
# =====================================================================

MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "IterativeMiniRocket": IterativeMiniRocket,
    "InceptionTime": InceptionTime,
    "TCN": TCN,
    "BiGRU": BiGRU,
}

def build_model(input_channels, num_classes, config):
    model_name = config.MODEL_NAME

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    ModelClass = MODEL_REGISTRY[model_name]

    return ModelClass(
        in_channels=input_channels, 
        num_classes=num_classes, 
        config=config,
        use_mel=getattr(config, 'USE_MEL', True)
    )