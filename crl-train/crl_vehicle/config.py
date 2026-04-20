from __future__ import annotations
from dataclasses import dataclass

LABEL_BACKGROUND = -1
LABEL_MULTI = -2
CATEGORY_TO_IDX = {"pedestrian": 0, "light": 1, "medium": 2, "heavy": 3}

DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm": ["light", "polaris"],
        "polaris0215pm": ["light", "polaris"],
        "polaris0235pm_nolineofsig": ["light", "polaris"],
        "warhog1135am": ["light", "warhog"],
        "warhog1149am": ["light", "warhog"],
        "warhog_nolineofsight": ["light", "warhog"],
        "silverado0255pm": ["heavy", "pickup"],
        "silverado0315pm": ["heavy", "pickup"],
    },
    "focal": {
        "walk": ["pedestrian", "walk"],
        "walk2": ["pedestrian", "walk"],
        "bicycle": ["pedestrian", "bicycle"],
        "bicycle2": ["pedestrian", "bicycle"],
        "motor": ["light", "motorcycle"],
        "motor2": ["light", "motorcycle"],
        "scooter": ["light", "scooter"],
        "scooter2": ["light", "scooter"],
        "forester": ["medium", "forester"],
        "forester2": ["medium", "forester"],
        "mustang": ["medium", "mustang"],
        "mustang0528": ["medium", "mustang"],
        "mustang2": ["medium", "mustang"],
        "pickup": ["heavy", "pickup"],
        "pickup2": ["heavy", "pickup"],
        "tesla": ["heavy", "tesla"],
        "tesla2": ["heavy", "tesla"],
    },
    "m3nvc": {
        "background": ["background", "background"],
        "cx30": ["medium", "cx30"],
        "miata": ["medium", "miata"],
        "mustang": ["medium", "mustang"],
        "gle350": ["heavy", "gle350"],
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

    # Frontend
    frontend_type: str = "multiscale"   # "multiscale" | "morlet"
    fused_seq_len: int = 32             # per-sensor token count after AdaptiveAvgPool1d
    morlet_kernel_size: int = 257
    multiscale_pool_stride: int = 16
    morlet_pool_stride: int = 64

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

    def modality_cfg(self, sensor: str) -> ModalityConfig:
        if sensor == "audio":
            return ModalityConfig(sample_rate=16000, window_size=16000, n_channels=1)
        elif sensor == "seismic":
            return ModalityConfig(sample_rate=200, window_size=200, n_channels=1)
        raise ValueError(f"Unknown modality: {sensor!r}")
