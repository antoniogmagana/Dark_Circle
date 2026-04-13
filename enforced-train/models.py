import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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
        self.bn1 = nn.BatchNorm2d(config.CHANNELS[0])
        self.conv2 = nn.Conv2d(
            config.CHANNELS[0],
            config.CHANNELS[1],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.bn2 = nn.BatchNorm2d(config.CHANNELS[1])
        self.conv3 = nn.Conv2d(
            config.CHANNELS[1],
            config.CHANNELS[2],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.bn3 = nn.BatchNorm2d(config.CHANNELS[2])
        self.conv4 = nn.Conv2d(
            config.CHANNELS[2],
            config.CHANNELS[3],
            kernel_size=config.KERNEL,
            padding=config.PADS,
        )
        self.bn4 = nn.BatchNorm2d(config.CHANNELS[3])
        # pool applied after conv1 and conv2; adaptive pool collapses the tail
        # regardless of input resolution (works for both audio and seismic)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.LazyLinear(config.HIDDEN)
        self.fc2 = nn.Linear(config.HIDDEN, num_classes)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_optimizer(self):
        return optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-4)


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
            nn.Conv1d(bottleneck_size, nb_filters, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm1d(nb_filters * (len(kernels) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        outs = [conv(bottleneck) for conv in self.conv_branches]
        outs.append(self.maxpool_branch(bottleneck))
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

        # Downsampling stem: normalises T to ~200 samples so INCEPTION_KERNELS remain
        # meaningful regardless of REF_SAMPLE_RATE. stride=1 at seismic-only rates.
        stem_stride = getattr(config, 'INCEPTION_STEM_STRIDE', 1)
        if stem_stride > 1:
            stem_k = 2 * stem_stride - 1   # odd kernel → symmetric padding
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=stem_k,
                          stride=stem_stride, padding=stem_stride - 1, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
            )
        else:
            self.stem = nn.Identity()

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
        x = self.stem(x)
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
# 4. MODEL REGISTRY
# =====================================================================

MODEL_REGISTRY = {
    "DetectionCNN": DetectionCNN,
    "ClassificationCNN": ClassificationCNN,
    "WaveformClassificationCNN": WaveformClassificationCNN,
    "ClassificationLSTM": ClassificationLSTM,
    "InceptionTime": InceptionTime,
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