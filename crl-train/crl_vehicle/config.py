from __future__ import annotations
from dataclasses import dataclass, field

LABEL_BACKGROUND = -1
LABEL_MULTI = -2
CATEGORY_TO_IDX = {"pedestrian": 0, "light": 1, "medium": 2, "heavy": 3}

DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm":             ["light", "polaris", "train"],
        "polaris0215pm":             ["light", "polaris", "val"],
        "polaris0235pm_nolineofsig": ["light", "polaris", "test"],
        "warhog1135am":              ["light", "warhog",  "train"],
        "warhog1149am":              ["light", "warhog",  "val"],
        "warhog_nolineofsight":      ["light", "warhog",  "test"],
        "silverado0255pm":           ["heavy", "pickup",  "train"],
        "silverado0315pm":           ["heavy", "pickup",  "split"],
    },
    "focal": {
        "walk":        ["pedestrian", "walk",       "train"],
        "walk2":       ["pedestrian", "walk",       "split"],
        "bicycle":     ["pedestrian", "bicycle",    "train"],
        "bicycle2":    ["pedestrian", "bicycle",    "split"],
        "motor":       ["light",      "motorcycle", "train"],
        "motor2":      ["light",      "motorcycle", "split"],
        "scooter":     ["light",      "scooter",    "train"],
        "scooter2":    ["light",      "scooter",    "split"],
        "forester":    ["medium",     "forester",   "train"],
        "forester2":   ["medium",     "forester",   "split"],
        "mustang":     ["medium",     "mustang",    "train"],
        "mustang2":    ["medium",     "mustang",    "val"],
        "mustang0528": ["medium",     "mustang",    "test"],
        "pickup":      ["heavy",      "pickup",     "train"],
        "pickup2":     ["heavy",      "pickup",     "split"],
        # "tesla":       ["heavy",      "tesla",      "train"],
        # "tesla2":      ["heavy",      "tesla",      "split"],
    },
    "m3nvc": {
        "background": ["background", "background"],
        "cx30":       ["medium", "cx30",    "split_runs"],
        "miata":      ["medium", "miata",   "split_runs"],
        "mustang":    ["medium", "mustang", "split_runs"],
        "gle350":     ["heavy",  "gle350",  "split_runs"],
    },
}


@dataclass
class ModalityConfig:
    # Defaults match the canonical seismic target (post-resample). The actual
    # values per-modality come from CRLConfig.modality_cfg(sensor) and override
    # these defaults; the dataclass defaults only matter if ModalityConfig() is
    # constructed without args (mostly in tests).
    sample_rate: int = 100
    window_size: int = 100
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
    # "multiscale"              — shared learned conv bank (early fusion).
    # "morlet"                  — shared Morlet bank per sensor, SR-derived freq range.
    # "morlet_per_sensor"       — Morlet bank per sensor with explicit freq ranges
    #                             from morlet_per_sensor_params (below).
    # "morlet_fused"            — Morlet bank per sensor (same per-sensor params
    #                             as morlet_per_sensor), then AdaptiveAvgPool1d
    #                             to fused_seq_len and time-concat → shared
    #                             encoder. Requires matching out_channels_frac.
    # "morlet_learnable"        — Late-fusion Morlet (like morlet_per_sensor)
    #                             with learnable scales in log-space. Optionally
    #                             learnable per-filter w0 via morlet_learnable_w0.
    #                             Initialized identically to morlet_per_sensor.
    # "morlet_learnable_fused"  — Early-fusion version of morlet_learnable
    #                             (matches morlet_fused topology).
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

    # Learnable Morlet variants (frontend_type ∈ {morlet_learnable,
    # morlet_learnable_fused}): scales are always learnable (parameterized
    # in log-space for positivity); per-filter w0 is learnable only when
    # morlet_learnable_w0=True. LR multiplier keeps the init-near-optimal
    # filterbank from wandering — learnable-filterbank literature uses
    # 0.1× backbone LR as the safe default.
    morlet_learnable_w0:       bool  = False
    morlet_learnable_lr_mult:  float = 0.1

    # Two-stage training (stage 2): when train.py --init-from-run loads a
    # converged fixed-Morlet checkpoint into a learnable model, this
    # multiplier reduces the encoder/decoder LR so filters are the primary
    # mover and the encoder just fine-tunes. Filter LR stays on
    # morlet_learnable_lr_mult. Only active in stage-2 runs.
    stage2_encoder_lr_mult:    float = 0.3

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

    # Per-sensor kernel-size lists for frontend_type="multiscale".
    #
    # Each Conv1D branch sees roughly one cycle of frequency f when its kernel
    # size ≈ SR / f. Sensors NOT in this dict fall through to
    # MultiScale1DFrontend's built-in default of [9, 19, 39] — i.e. omit a
    # sensor to keep the existing behavior unchanged.
    #
    # Audio default below adds ks=159 to the standard ladder so the audio
    # branch can see ~one cycle at ~100 Hz (engine-fundamental band) at
    # SR=16000. The previous max kernel of 39 only reached ~400 Hz.
    #
    # Constraints (validated in __post_init__):
    #   - Every kernel size MUST be odd (Conv1D padding=ks//2 is symmetric
    #     only for odd ks; even ks gives an off-by-one shift).
    #   - Every kernel size MUST be ≤ window_size for that sensor. A kernel
    #     longer than the input is a degenerate global filter.
    #
    # Maximum viable kernel (one-cycle frequency floor) at default SR/window:
    #   sensor    SR (Hz)   window   ks_max   one-cycle freq @ ks_max
    #   ----------------------------------------------------------------
    #   audio     16000     16000    15999     ~1.0 Hz
    #   seismic     100       100       99     ~1.0 Hz
    # Practically, kernels approaching window size give no localization;
    # treat ks ≈ window/2 as the soft ceiling for keeping multiple
    # receptive fields per token.
    multiscale_kernel_sizes: dict = field(default_factory=lambda: {
        "audio": [9, 19, 39, 159],
        # seismic intentionally absent → uses [9, 19, 39] default,
        # which already covers ~2.5 Hz–11 Hz at SR=100.
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
    # ID split schema (opt-in). When True, train.py reads
    # train/val/test assignments from DATASET_VEHICLE_MAP rather than
    # from on-disk directory layout. See
    # docs/superpowers/specs/2026-04-25-id-split-schema-design.md.
    use_id_split: bool = False

    # Stratified partner sampling
    n_partners_same_type: int = 1
    n_partners_diff_type: int = 1
    n_partners_cross_ds: int = 1

    # Contrastive mode (training_mode="contrastive")
    contrastive_temperature: float = 0.1
    contrastive_d_proj: int = 64

    # Disentangled mode (training_mode="disentangled")
    # Two-block latent: z[0:d_signal] is the labeled "vehicle signal" subspace,
    # z[d_signal:d_z] is environment/noise. Routing is enforced by losses
    # (cross-modal alignment, env temporal stability, signal intervention
    # invariance), not by per-feature dim assignment. Aux pres+type heads
    # read the full d_signal block.
    d_signal: int = 12
    lambda_align:      float = 1.0   # cross-modal coherence on z_signal (per-sensor only)
    lambda_stab:       float = 0.1   # env temporal stability on consecutive partners
    lambda_interv_inv: float = 1.0   # z_signal invariance under noise interventions

    def __post_init__(self) -> None:
        for sensor, ks_list in self.multiscale_kernel_sizes.items():
            window = self.modality_cfg(sensor).window_size
            for ks in ks_list:
                if ks % 2 != 1:
                    raise ValueError(
                        f"multiscale_kernel_sizes[{sensor!r}] contains even "
                        f"kernel {ks}; Conv1D padding=ks//2 requires odd ks."
                    )
                if ks > window:
                    raise ValueError(
                        f"multiscale_kernel_sizes[{sensor!r}] kernel {ks} "
                        f"exceeds {sensor} window_size {window}; "
                        f"max viable is {window if window % 2 == 1 else window - 1}."
                    )

    def modality_cfg(self, sensor: str) -> ModalityConfig:
        """Canonical (post-resample) sample rate and window size per sensor.

        Targets chosen to match expected production hardware:
          - audio   = 16000 Hz (focal/iobt native; m3nvc upsampled 1600→16000)
          - seismic = 100 Hz   (focal/iobt native; m3nvc downsampled 200→100)

        The dataset loader resamples every file to these canonical rates before
        windowing — see crl_vehicle/data/dataset.py:_SOURCE_RATES and
        _resample_to_target. Frontends and Morlet bands assume these canonical
        rates, so do not change them without retraining.

        m3nvc seismic downsample is safe (torchaudio.resample anti-aliases) and
        loses only the 50-100 Hz band, well above the 2-40 Hz Morlet seismic
        range in morlet_per_sensor_params.
        """
        if sensor == "audio":
            return ModalityConfig(sample_rate=16000, window_size=16000, n_channels=1)
        elif sensor == "seismic":
            return ModalityConfig(sample_rate=100, window_size=100, n_channels=1)
        raise ValueError(f"Unknown modality: {sensor!r}")
