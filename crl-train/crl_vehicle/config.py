"""
CRLConfig — single source of truth for all hyperparameters.

Design principle: audio and seismic are processed completely independently
through separate SpectrogramFrontend → SSM → CausalEncoder → SCM chains.
They share the same d_z latent structure (same causal variables) but have
distinct signal processing parameters. Late fusion only happens at the
downstream head.

Per-modality parameters live in ModalityConfig; shared architecture and
training parameters live in CRLConfig.

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
        "polaris0150pm":              "light",
        "polaris0215pm":              "light",
        "polaris0235pm_nolineofsig":  "light",
        "warhog1135am":               "light",
        "warhog1149am":               "light",
        "warhog_nolineofsight":       "light",
        "silverado0255pm":            "utility",
        "silverado0315pm":            "utility",
    },
    "focal": {
        "walk":        "pedestrian",
        "walk2":       "pedestrian",
        "bicycle":     "pedestrian",
        "bicycle2":    "pedestrian",
        "motor":       "light",
        "motor2":      "light",
        "scooter":     "light",
        "scooter2":    "light",
        "forester":    "utility",
        "forester2":   "utility",
        "mustang":     "sport",
        "mustang0528": "sport",
        "mustang2":    "sport",
        "pickup":      "utility",
        "pickup2":     "utility",
        "tesla":       "sport",
        "tesla2":      "sport",
    },
    "m3nvc": {
        "background":    "background",
        "cx30":          "utility",
        "miata":         "sport",
        "mustang":       "sport",
        "gle350":        "utility",
        "cx30_miata":    "multi",
        "cx30_mustang":  "multi",
        "miata_mustang": "multi",
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

    # SSM (shared across modalities — both produce T'=25)
    d_model:         int   = 64     # feature dimension in/out of SSM
    ssm_nhead:       int   = 4      # Transformer attention heads
    ssm_layers:      int   = 2      # Transformer encoder layers
    ssm_dropout:     float = 0.1

    # Encoder latent dims (same causal variables for both modalities)
    d_z_presence:    int   = 1      # is vehicle present?
    d_z_type:        int   = 4      # vehicle type (# classes)
    d_z_proximity:   int   = 1      # scalar proximity
    d_z_noise:       int   = 4      # unstructured nuisance
    # total d_z = 10

    # SCM
    scm_hidden:      int   = 32

    # Whether the SSM + CausalEncoder weights are shared across modalities.
    # False = separate weights per modality (default: more capacity, less transfer).
    # True  = shared weights (stronger inductive bias that both sensors encode
    #         the same causal structure).
    share_encoder:   bool  = False

    # Downstream fusion strategy for detection/classification heads:
    #   "concat"  — concatenate z_audio and z_seismic → 2*d_z features
    #   "vote"    — separate heads per modality, average probabilities
    #   "any"     — use whichever modality is available (no fusion)
    fusion:          str   = "vote"

    # Training
    batch_size:           int   = 64
    lr:                   float = 1e-3
    lr_min:               float = 1e-5
    wd:                   float = 1e-4
    warmup_epochs:        int   = 5
    cosine_period:        int   = 20
    n_epochs:             int   = 100
    num_workers:          int   = 8
    early_stop_patience:  int   = 10

    # Loss weights
    beta_start:           float = 0.0
    beta_end:             float = 4.0
    beta_anneal_epochs:   int   = 20
    lambda_causal:        float = 5.0
    lambda_interv:        float = 2.0
    lambda_disent:        float = 0.5
    lambda_task:          float = 1.0
    lambda_l1_graph:             float = 0.01   # L1 sparsity on SCM adjacency weights
    lambda_l1_graph_anneal_epochs: int = 20    # ramp 0 → lambda_l1_graph over this many epochs

    # Curriculum
    unknown_interv_start_epoch: int = 10
    unknown_interv_ramp_epochs: int = 10

    # Multi-horizon pair construction
    n_horizons:         int   = 10    # n ∈ {1..n_horizons} for temporal pairs
    horizon_stride_sec: float = 0.1   # seconds between successive x(t) anchor windows

    # All-interventions contrast loss weights
    lambda_contrast: float = 2.0   # invariance of vehicle dims across interventions
    lambda_equiv:    float = 0.5   # equivariance: noise dims must vary across interventions
    lambda_collapse: float = 1.0   # posterior collapse penalty

    # Training throughput
    steps_per_epoch: int | None = None  # cap gradient steps per epoch (None = full epoch)

    # Paths
    save_dir:        str   = "saved_crl"

    @property
    def d_z(self) -> int:
        return self.d_z_presence + self.d_z_type + self.d_z_proximity + self.d_z_noise

    def modality_cfg(self, modality: str) -> ModalityConfig:
        if modality == "audio":
            return self.audio_cfg
        if modality == "seismic":
            return self.seismic_cfg
        raise ValueError(f"Unknown modality: {modality!r}")
