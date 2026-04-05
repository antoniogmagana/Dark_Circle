"""
Shared constants for the CRL training pipeline.
Ported from model-train/config.py — no interactive input() calls.
"""

# =====================================================================
# Sample rates
# =====================================================================

NATIVE_SR = {
    "iobt":  {"audio": 16000, "seismic": 100,  "accel": 100},
    "focal": {"audio": 16000, "seismic": 100,  "accel": 100},
    "m3nvc": {"audio": 1600,  "seismic": 200,  "accel": 200},
}

REF_SR = 16000          # all modalities resampled to this
SAMPLE_SECONDS = 1.0    # window length

# =====================================================================
# Label spaces
# =====================================================================

# 4-class vehicle category map (indices used by ClassificationHead)
CLASS_MAP = {0: "pedestrian", 1: "light", 2: "sport", 3: "utility"}
CLASS_NAMES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]  # ordered list

# Inverse map: category string → class index
CATEGORY_TO_IDX = {v: k for k, v in CLASS_MAP.items()}

# Special label sentinels
LABEL_BACKGROUND = -1   # m3nvc background recordings
LABEL_MULTI      = -2   # ambiguous multi-vehicle m3nvc recordings

# =====================================================================
# Vehicle → category mapping
# Ported verbatim from model-train/config.py DATASET_VEHICLE_MAP.
# Returns category string; "background" and "multi" are handled specially.
# =====================================================================

DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm":           "light",
        "polaris0215pm":           "light",
        "polaris0235pm_nolineofsig": "light",
        "warhog1135am":            "light",
        "warhog1149am":            "light",
        "warhog_nolineofsight":    "light",
        "silverado0255pm":         "utility",
        "silverado0315pm":         "utility",
    },
    "focal": {
        "walk":       "pedestrian",
        "walk2":      "pedestrian",
        "bicycle":    "pedestrian",
        "bicycle2":   "pedestrian",
        "motor":      "light",
        "motor2":     "light",
        "scooter":    "light",
        "scooter2":   "light",
        "forester":   "utility",
        "forester2":  "utility",
        "mustang":    "sport",
        "mustang0528":"sport",
        "mustang2":   "sport",
        "pickup":     "utility",
        "pickup2":    "utility",
        "tesla":      "sport",
        "tesla2":     "sport",
    },
    "m3nvc": {
        "background":    "background",
        "cx30":          "utility",
        "miata":         "sport",
        "mustang":       "sport",
        "gle350":        "utility",
        # Multi-vehicle (ambiguous) — excluded from classification training
        "cx30_miata":    "multi",
        "cx30_mustang":  "multi",
        "miata_mustang": "multi",
    },
}

# =====================================================================
# ADC normalisation
# =====================================================================

BIT_DEPTH_MAP = {"audio": 16, "seismic": 24, "accel": 24}
ADC_SCALE     = {s: 2 ** (b - 1) for s, b in BIT_DEPTH_MAP.items()}

# =====================================================================
# Model architecture defaults
# =====================================================================

Z_VEH_DIM           = 32
Z_ENV_DIM           = 16
MODALITY_FEATURE_DIM = 128   # per-encoder output dim before fusion

MODALITIES = ["audio", "seismic", "accel"]

# Channels per modality
MODALITY_CHANNELS = {"audio": 1, "seismic": 1, "accel": 3}

# =====================================================================
# Training defaults
# =====================================================================

BATCH_SIZE          = 128
CRL_EPOCHS          = 150
DOWNSTREAM_EPOCHS   = 50
LEARNING_RATE       = 1e-3
LAMBDA_SLOW         = 5.0
BETA_KL             = 1.0
NUM_WORKERS         = 4
EARLY_STOP_PATIENCE = 8
LR_FACTOR           = 0.5
LR_PATIENCE         = 3
