"""
CRLConfig — single source of truth for all hyperparameters.

Design principle: audio and seismic are processed completely independently
through separate SpectrogramFrontend → SSM → MultiTaskEncoder chains.
Each modality produces three task-specific embeddings (presence, type,
instance) via separate projection branches, eliminating gradient competition
between differently-sized latent blocks.

Audio target SR  : 16000 Hz → window_size=16000, n_fft=512, hop_length=160
                              → T' = 16000 // 160 + 1 = 101 frames
                              → n_freq_bins = n_mels = 64
Seismic target SR: 200 Hz   → window_size=200, n_fft=64, hop_length=8
                              → T' = 200 // 8 + 1 = 26 frames
                              → n_freq_bins = n_fft//2+1 = 33
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Vehicle label spaces
# ---------------------------------------------------------------------------

CLASS_MAP = {0: "pedestrian", 1: "light", 2: "sport", 3: "utility"}
CATEGORY_TO_IDX = {v: k for k, v in CLASS_MAP.items()}
LABEL_BACKGROUND = -1   # background recordings (m3nvc)
LABEL_MULTI      = -2   # ambiguous multi-vehicle recordings

INSTANCE_MAP = {
    0: "polaris", 1: "warhog", 2: "pickup",
    3: "walk", 4: "bicycle", 5: "motorcycle",
    6: "scooter", 7: "forester", 8: "mustang",
    9: "ev", 10: "cx30", 11: "miata", 12: "gle350",
}
INSTANCE_TO_IDX = {v: k for k, v in INSTANCE_MAP.items()}
LABEL_INSTANCE_BACKGROUND = -1
LABEL_INSTANCE_MULTI      = -2

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
        "silverado0255pm":            ["utility", "pickup"],
        "silverado0315pm":            ["utility", "pickup"],
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
        "forester":    ["utility",    "forester"],
        "forester2":   ["utility",    "forester"],
        "mustang":     ["sport",      "mustang"],
        "mustang0528": ["sport",      "mustang"],
        "mustang2":    ["sport",      "mustang"],
        "pickup":      ["utility",    "pickup"],
        "pickup2":     ["utility",    "pickup"],
        "tesla":       ["sport",      "ev"],
        "tesla2":      ["sport",      "ev"],
    },
    "m3nvc": {
        "background":    ["background", "background"],
        "cx30":          ["utility",    "cx30"],
        "miata":         ["sport",      "miata"],
        "mustang":       ["sport",      "mustang"],
        "gle350":        ["utility",    "gle350"],
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
    """
    Signal processing parameters for a single sensor modality.
    Audio uses a mel spectrogram; seismic uses a linear-scale STFT.
    """
    sample_rate:  int   = 200    # target SR after resampling
    window_size:  int   = 200    # samples per 1-second window (= sample_rate × 1s)
    n_channels:   int   = 1      # audio=1, seismic=1
    n_fft:        int   = 64     # STFT window size in samples
    hop_length:   int   = 8      # STFT frame hop in samples
    n_mels:       int   = 0      # >0 → mel spectrogram; 0 → linear STFT (seismic)
    f_min:        float = 0.0    # mel lower bound in Hz (audio only)
    f_max:        float = 0.0    # mel upper bound in Hz (audio only; 0 = sr/2)

    @property
    def t_prime(self) -> int:
        """Temporal frames produced by STFT with center=True padding."""
        return self.window_size // self.hop_length + 1

    @property
    def n_freq_bins(self) -> int:
        """Frequency bins output by the spectrogram frontend."""
        if self.n_mels > 0:
            return self.n_mels
        return self.n_fft // 2 + 1

    @property
    def filterbank_out_channels(self) -> int:
        """Total feature channels fed into TemporalSSM (freq_bins × n_channels)."""
        return self.n_freq_bins * self.n_channels


def default_audio_config() -> ModalityConfig:
    """
    Audio at 16 kHz target SR — mel spectrogram frontend.
    1 second = 16000 samples; n_fft=512 (32 ms), hop=160 (10 ms).
    T' = 16000 // 160 + 1 = 101 frames.
    64 mel bins, 50 Hz – 8 kHz covers vehicle acoustic signatures.
    """
    return ModalityConfig(
        sample_rate=16000,
        window_size=16000,
        n_channels=1,
        n_fft=512,
        hop_length=160,
        n_mels=64,
        f_min=50.0,
        f_max=8000.0,
    )


def default_seismic_config() -> ModalityConfig:
    """
    Seismic at 200 Hz target SR — linear STFT frontend.
    1 second = 200 samples; n_fft=64 (320 ms), hop=8 (40 ms).
    T' = 200 // 8 + 1 = 26 frames; 33 linear frequency bins (0–100 Hz).
    Linear scale is appropriate: mel compression distorts the low-frequency
    vehicle vibration bands (2–90 Hz) that contain diagnostic information.
    """
    return ModalityConfig(
        sample_rate=200,
        window_size=200,
        n_channels=1,
        n_fft=64,
        hop_length=8,
        n_mels=0,
        f_min=0.0,
        f_max=0.0,
    )


# ---------------------------------------------------------------------------
# CRLConfig
# ---------------------------------------------------------------------------

@dataclass
class CRLConfig:
    """
    Full pipeline configuration.
    Modality-specific signal processing lives in audio_cfg / seismic_cfg.
    Audio produces T'=101, seismic T'=26. share_encoder=False (default)
    gives each modality independent SSM/encoder weights.
    """

    # Per-modality signal processing configs
    audio_cfg:   ModalityConfig = field(default_factory=default_audio_config)
    seismic_cfg: ModalityConfig = field(default_factory=default_seismic_config)

    # Data
    sample_seconds:  float = 1.0    # window duration in seconds

    # SSM (shared across modalities)
    d_model:         int   = 64     # feature dimension in/out of SSM
    ssm_nhead:       int   = 4      # Transformer attention heads
    ssm_layers:      int   = 2      # Transformer encoder layers
    ssm_dropout:     float = 0.1

    # Task-specific embedding dimensions (separate branch per task)
    # Larger dims give the encoder more capacity per task without cross-task
    # gradient competition through a shared projection.
    d_pres: int = 16    # presence embedding (binary: vehicle present/absent)
    d_type: int = 32    # type embedding (4 classes: pedestrian/light/sport/utility)
    d_inst: int = 64    # instance embedding (13 classes)

    # Whether the SSM + encoder weights are shared across modalities.
    share_encoder:   bool  = False

    # Downstream fusion strategy for inference:
    #   "vote"    — separate heads per modality, average probabilities
    #   "any"     — use whichever modality is available
    fusion:          str   = "vote"

    # Training
    batch_size:           int   = 64
    lr:                   float = 1e-3
    lr_min:               float = 1e-4
    wd:                   float = 1e-4
    warmup_epochs:        int   = 5
    n_epochs:             int   = 100
    num_workers:          int   = 60
    early_stop_patience:  int   = 25

    # Loss weights
    lambda_type:   float = 2.0    # weight on vehicle type CE loss
    lambda_inst:   float = 1.0    # weight on instance CE loss
    lambda_recon:  float = 0.1    # weight on reconstruction regularizer
    lambda_tc:     float = 0.5    # weight on total-correlation disentanglement penalty
                                  # applied to e_type and e_pres to reduce z_veh noise leakage

    # Data windowing (controls sliding-window stride in SensorDataset)
    horizon_stride_sec: float = 0.1   # seconds between successive anchor windows
    n_horizons:         int   = 10    # max horizon n for MultiHorizonPairDataset (unused in training)

    # Training throughput
    steps_per_epoch: int | None = None  # cap gradient steps per epoch (None = full epoch)

    # Paths
    save_dir:        str   = "saved_crl"

    @property
    def d_embed(self) -> int:
        """Total concatenated embedding dimension (for decoder input)."""
        return self.d_pres + self.d_type + self.d_inst

    def modality_cfg(self, modality: str) -> ModalityConfig:
        if modality == "audio":
            return self.audio_cfg
        if modality == "seismic":
            return self.seismic_cfg
        raise ValueError(f"Unknown modality: {modality!r}")
