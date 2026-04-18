"""
CRLConfig — single source of truth for all hyperparameters.

Design principle: audio and seismic are processed completely independently
through separate frontend → encoder chains, eliminating gradient competition
between differently-sized latent blocks.

Audio target SR  : 16000 Hz (window_size=16000)
Seismic target SR: 200 Hz   (window_size=200)
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Vehicle label spaces
# ---------------------------------------------------------------------------

CLASS_MAP = {0: "pedestrian", 1: "light", 2: "sport", 3: "utility"}
CATEGORY_TO_IDX = {v: k for k, v in CLASS_MAP.items()}
LABEL_BACKGROUND = -1   # background recordings (m3nvc)
LABEL_MULTI      = -2   # ambiguous multi-vehicle recordings

# Native sample rates per dataset and sensor modality
NATIVE_SR = {
    "iobt":  {"audio": 16000, "seismic": 100,  "accel": 100},
    "focal": {"audio": 16000, "seismic": 100,  "accel": 100},
    "m3nvc": {"audio": 1600,  "seismic": 200,  "accel": 200},
}

# ADC normalisation: amplitude / (2 ** (bit_depth - 1))
ADC_SCALE = {"audio": 32768.0, "seismic": 8388608.0, "accel": 8388608.0}

# Vehicle → category string per dataset
DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm":              ["light",   "polaris"],
        "polaris0215pm":              ["light",   "polaris"],
        "polaris0235pm_nolineofsig":  ["light",   "polaris"],
        "warhog1135am":               ["light",   "warhog"],
        "warhog1149am":               ["light",   "warhog"],
        "warhog_nolineofsight":       ["light",   "warhog"],
        "silverado0255pm":            ["heavy", "pickup"],
        "silverado0315pm":            ["heavy", "pickup"],
    },
    "focal": {
        "walk":        ["pedestrian", "walk"],
        "walk2":       ["pedestrian", "walk"],
        "bicycle":     ["pedestrian", "bicycle"],
        "bicycle2":    ["pedestrian", "bicycle"],
        "motor":       ["light",      "motorcycle"],
        "motor2":      ["light",      "motorcycle"],
        "scooter":     ["light",      "scooter"],
        "scooter2":    ["light",      "scooter"],
        "forester":    ["medium",    "forester"],
        "forester2":   ["medium",    "forester"],
        "mustang":     ["medium",      "mustang"],
        "mustang0528": ["medium",      "mustang"],
        "mustang2":    ["medium",      "mustang"],
        "pickup":      ["heavy",    "pickup"],
        "pickup2":     ["heavy",    "pickup"],
        "tesla":       ["heavy",      "ev"],
        "tesla2":      ["heavy",      "ev"],
    },
    "m3nvc": {
        "background":    ["background", "background"],
        "cx30":          ["medium",    "cx30"],
        "miata":         ["medium",      "miata"],
        "mustang":       ["medium",      "mustang"],
        "gle350":        ["heavy",    "gle350"],
        "cx30_miata":    ["multi",      "multi"],
        "cx30_mustang":  ["multi",      "multi"],
        "miata_mustang": ["multi",      "multi"],
    },
}

MODALITIES = ["audio", "seismic"]


# ---------------------------------------------------------------------------
# Per-modality signal processing parameters
# ---------------------------------------------------------------------------

@dataclass
class ModalityConfig:
    """Signal processing parameters for a single sensor modality."""
    sample_rate:  int   = 200    # target SR after resampling
    window_size:  int   = 200    # samples per 1-second window (= sample_rate × 1s)
    n_channels:   int   = 1      # audio=1, seismic=1


def default_audio_config() -> ModalityConfig:
    return ModalityConfig(sample_rate=16000, window_size=16000, n_channels=1)


def default_seismic_config() -> ModalityConfig:
    return ModalityConfig(sample_rate=200, window_size=200, n_channels=1)


# ---------------------------------------------------------------------------
# CRLConfig
# ---------------------------------------------------------------------------

@dataclass
class CRLConfig:
    """
    Full pipeline configuration.
    Modality-specific signal processing lives in audio_cfg / seismic_cfg.
    """

    # Per-modality signal processing configs
    audio_cfg:   ModalityConfig = field(default_factory=default_audio_config)
    seismic_cfg: ModalityConfig = field(default_factory=default_seismic_config)

    # Data
    sample_seconds:  float = 1.0    # window duration in seconds

    # Encoder / decoder
    d_model:         int   = 64     # internal feature dimension
    n_heads:         int   = 4      # Transformer attention heads
    n_layers:        int   = 2      # Transformer encoder layers

    # Training
    batch_size:           int   = 512
    lr:                   float = 3e-4
    lr_min:               float = 1e-4
    wd:                   float = 1e-4
    n_epochs:             int   = 100
    num_workers:          int   = 12
    early_stop_patience:  int   = 25

    # Loss weights
    lambda_interv:    float = 1.0    # weight on intervention matching loss

    # Adaptive beta schedule (Options 1 + 3)
    beta_step:        float = 0.02   # amount to increment/decrement beta each epoch
    kl_floor:         float = 0.01   # raw KL below this → decay beta (collapse guard)
    kl_target:        float = 0.5    # raw KL above this → raise beta unconditionally
    recon_min_delta:  float = 0.005  # min improvement in val_recon to count as "improving"

    # Data windowing (controls sliding-window stride in SensorDataset)
    horizon_stride_sec: float = 0.7   # seconds between successive anchor windows

    # Training throughput
    steps_per_epoch: int | None = None  # cap training batches per epoch (None = full epoch)

    # Paths
    save_dir:        str   = "saved_crl"
    cache_dir:       str   = "saved_crl/cache"

    # Frontend architecture
    frontend_type:          str = "multiscale"
    morlet_kernel_size:     int = 257
    morlet_pool_stride:     int = 64   # AvgPool1d stride after Morlet frontend (16000 → 250 frames for audio)
    multiscale_pool_stride: int = 16   # AvgPool1d stride after MultiScale frontend (4000 → 250 frames for audio)

    def modality_cfg(self, modality: str) -> ModalityConfig:
        if modality == "audio":
            return self.audio_cfg
        if modality == "seismic":
            return self.seismic_cfg
        raise ValueError(f"Unknown modality: {modality!r}")
