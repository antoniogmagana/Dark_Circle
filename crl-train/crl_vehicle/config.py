from __future__ import annotations
from dataclasses import dataclass, field

LABEL_BACKGROUND = -1
LABEL_MULTI = -2
CATEGORY_TO_IDX = {"pedestrian": 0, "light": 1, "medium": 2, "heavy": 3}

DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm": ["light", "polaris", "test"],
        "polaris0215pm": ["light", "polaris", "test"],
        "polaris0235pm_nolineofsig": ["light", "polaris", "test"],
        "warhog1135am": ["light", "warhog", "test"],
        "warhog1149am": ["light", "warhog", "test"],
        "warhog_nolineofsight": ["light", "warhog", "test"],
        "silverado0255pm": ["heavy", "pickup", "test"],
        "silverado0315pm": ["heavy", "pickup", "test"],
    },
    "focal": {
        "walk": ["pedestrian", "walk", "train"],
        "walk2": ["pedestrian", "walk", "split"],
        "bicycle": ["pedestrian", "bicycle", "test"],
        "bicycle2": ["pedestrian", "bicycle", "split"],
        "motor": ["light", "motorcycle", "train"],
        "motor2": ["light", "motorcycle", "split"],
        "scooter": ["light", "scooter", "train"],
        "scooter2": ["light", "scooter", "train"],
        "forester": ["medium", "forester", "train"],
        "forester2": ["medium", "forester", "train"],
        "mustang": ["medium", "mustang", "train"],
        "mustang0528": ["medium", "mustang", "val"],
        "mustang2": ["medium", "mustang", "test"],
        "pickup": ["heavy", "pickup", "train"],
        "pickup2": ["heavy", "pickup", "val"],
        "tesla": ["heavy", "tesla", "train"],
        "tesla2": ["heavy", "tesla", "split"],
    },
    "m3nvc": {
        "background": ["background", "background"],
        "cx30": ["medium", "cx30", "train"],
        "miata": ["medium", "miata", "train"],
        "mustang": ["medium", "mustang", "train"],
        "gle350": ["heavy", "gle350", "train"],
    },
}

@dataclass
class ModalityConfig:
    sample_rate: int = 200
    window_size: int = 200
    n_channels: int = 1


@dataclass
class CRLConfig:
    # Latent space
    d_z: int = 24

    # Encoder/decoder
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2

    # Training mode — selects which TrainingMode + Prior to instantiate.
    # ("vae", "standard")    — classical CRL (Checkpoint 1).
    # ("vae", "conditional") — iVAE (Checkpoint 2, not yet implemented).
    # ("contrastive", *)     — NT-Xent (Checkpoint 3, not yet implemented).
    training_mode: str = "vae"
    prior_type:    str = "standard"

    # Frontend
    # "multiscale"        — shared learned conv bank (early fusion).
    # "morlet"            — shared Morlet bank per sensor, SR-derived freq range.
    # "morlet_per_sensor" — Morlet bank per sensor with explicit freq ranges
    #                       from morlet_per_sensor_params (below).
    frontend_type: str = "multiscale"
    fused_seq_len: int = 32             # per-sensor token count after AdaptiveAvgPool1d
    morlet_kernel_size: int = 257
    multiscale_pool_stride: int = 16
    morlet_pool_stride: int = 64

    # Phase channels: when True, Morlet variants emit [log_power, cos_phase,
    # sin_phase] → 3× channel count. Preserves phase/onset structure that
    # vehicle-onset discrimination depends on. Default False for backward
    # compatibility.
    morlet_use_phase: bool = False

    # Per-sensor Morlet frequency ranges for frontend_type="morlet_per_sensor".
    # Audio: 20 Hz–8 kHz (SR/2 band above speech; engine harmonics + tire noise).
    # Seismic: 2–40 Hz (typical vehicle ground-vibration band).
    # out_channels_frac scales d_model to produce the per-sensor channel count;
    # keep at 1.0 unless you want one sensor to dominate channel budget.
    morlet_per_sensor_params: dict = field(default_factory=lambda: {
        "audio":   {"freq_min": 20.0,  "freq_max": 8000.0,
                    "out_channels_frac": 1.0, "w0": 6.0,
                    "target_tokens": 32, "receptive_cycles": 3.0},
        "seismic": {"freq_min": 2.0,   "freq_max": 40.0,
                    "out_channels_frac": 1.0, "w0": 6.0,
                    "target_tokens": 32, "receptive_cycles": 3.0},
    })

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    lr_min: float = 1e-4
    wd: float = 1e-4
    n_epochs: int = 100
    num_workers: int = 24
    early_stop_patience: int = 25

    # Loss weights
    lambda_interv: float = 1.0
    lambda_aux_pres: float = 1.0
    lambda_aux_type: float = 1.0
    lambda_aux_prox: float = 0.1

    # Adaptive beta schedule
    beta_step: float = 0.02
    kl_floor: float = 0.01
    kl_target: float = 0.5
    recon_min_delta: float = 1e-4

    # Data
    horizon_stride_sec: float = 0.7

    # Stratified partner sampling
    n_partners_same_type: int = 1
    n_partners_diff_type: int = 1
    n_partners_cross_ds: int = 1

    # Contrastive mode (training_mode="contrastive")
    contrastive_temperature: float = 0.1
    contrastive_d_proj: int = 64

    def modality_cfg(self, sensor: str) -> ModalityConfig:
        if sensor == "audio":
            return ModalityConfig(sample_rate=16000, window_size=16000, n_channels=1)
        elif sensor == "seismic":
            return ModalityConfig(sample_rate=200, window_size=200, n_channels=1)
        raise ValueError(f"Unknown modality: {sensor!r}")
