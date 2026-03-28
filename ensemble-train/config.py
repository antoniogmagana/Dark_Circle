import os
import math
import random
import json
import torch
import numpy as np
from datetime import datetime


# =====================================================================
# 1. HARDWARE & DEVICE
# =====================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# =====================================================================
# 2. DATABASE PARAMETERS
# =====================================================================

DB_CONN_PARAMS = {
    "dbname": "lvc_db",
    "user": "lvc_toolkit",
    "password": os.environ.get("DB_PASSWORD"),
    "host": "localhost",
    "port": 5432,
}
if not DB_CONN_PARAMS["password"]:
    DB_CONN_PARAMS["password"] = input("Enter Database Password: ")

# =====================================================================
# 3. TRAINING MODE & SENSOR
# =====================================================================

TRAINING_MODE = os.environ.get("TRAINING_MODE")
if not TRAINING_MODE:
    TRAINING_MODE = input(
        'Enter Training Mode ["detection", "category", "instance"]: '
    )

ALL_SENSORS = ["audio", "seismic", "accel"]

TRAIN_SENSOR = os.environ.get("TRAIN_SENSOR")
if not TRAIN_SENSOR:
    TRAIN_SENSOR = input('Enter Sensor ["audio", "seismic", "accel"]: ')

TRAIN_SENSORS = [TRAIN_SENSOR]
IN_CHANNELS = 3 if TRAIN_SENSOR == "accel" else 1

# Sensors used for time-window alignment.  All sensors in this list
# must be present for a group to be valid.  This ensures that training
# audio alone and seismic alone produce identical sample lists.
ALIGN_SENSORS = ["audio", "seismic"]

INSTANCE_SEED = 0

# =====================================================================
# 4. DATASET & CLASS CONSTANTS
# =====================================================================

TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]
ACOUSTIC_SR = 16000

NATIVE_SR = {
    "iobt":  {"audio": 16000, "seismic": 100, "accel": 100},
    "focal": {"audio": 16000, "seismic": 100, "accel": 100},
    "m3nvc": {"audio": 1600,  "seismic": 200, "accel": 200},
}

REF_SAMPLE_RATE = max(NATIVE_SR[ds][TRAIN_SENSOR] for ds in TRAIN_DATASETS)

CLASS_MAP = {0: "pedestrian", 1: "light", 2: "sport", 3: "utility"}

DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm": "light",
        "polaris0215pm": "light",
        "polaris0235pm_nolineofsig": "light",
        "warhog1135am": "light",
        "warhog1149am": "light",
        "warhog_nolineofsight": "light",
        "silverado0255pm": "utility",
        "silverado0315pm": "utility",
    },
    "focal": {
        "walk": "pedestrian",
        "walk2": "pedestrian",
        "bicycle": "pedestrian",
        "bicycle2": "pedestrian",
        "motor": "light",
        "motor2": "light",
        "scooter": "light",
        "scooter2": "light",
        "forester": "utility",
        "forester2": "utility",
        "mustang": "sport",
        "mustang0528": "sport",
        "mustang2": "sport",
        "pickup": "utility",
        "pickup2": "utility",
        "tesla": "sport",
        "tesla2": "sport",
    },
    "m3nvc": {
        "background": "background",
        "cx30": "utility",
        "miata": "sport",
        "mustang": "sport",
        "miata_mustang": "sport",
        "gle350": "utility",
    },
}

# =====================================================================
# 5. DYNAMIC LABEL SPACE
# =====================================================================

ALL_INSTANCES = sorted({
    name for ds_map in DATASET_VEHICLE_MAP.values() for name in ds_map.keys()
})

random.seed(INSTANCE_SEED)
shuffled_instances = ALL_INSTANCES.copy()
random.shuffle(shuffled_instances)
INSTANCE_TO_CLASS = {name: idx for idx, name in enumerate(shuffled_instances)}

if TRAINING_MODE == "detection":
    NUM_CLASSES = 2
    CLASS_WEIGHTS = [25.5, 1.0]
elif TRAINING_MODE == "category":
    NUM_CLASSES = len(CLASS_MAP)
    CLASS_WEIGHTS = [28.0, 2.0, 13.0, 15.5]
elif TRAINING_MODE == "instance":
    NUM_CLASSES = len(INSTANCE_TO_CLASS)
    CLASS_WEIGHTS = [1.0] * NUM_CLASSES
else:
    raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")

# =====================================================================
# 6. MODEL SELECTION & VERSIONING
# =====================================================================

MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME = input("Enter Model Name: ")

RUN_ID = os.environ.get("RUN_ID")
if not RUN_ID:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_DIR = os.path.join("saved_models", TRAINING_MODE, MODEL_NAME, RUN_ID)
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
META_SAVE_PATH = os.path.join(RUN_DIR, "meta.pt")
IMG_SAVE_PATH = os.path.join(RUN_DIR, "conf_matrix.png")
JSON_LOG_PATH = os.path.join(RUN_DIR, "hyperparameters.json")
METRICS_LOG_PATH = os.path.join(RUN_DIR, "metrics.csv")

# =====================================================================
# 7. CORE TRAINING HYPERPARAMETERS
# =====================================================================

BATCH_SIZE = 128
EPOCHS = 50
NUM_WORKERS = 32
LOG_INTERVAL = 10

BEST_MODEL_METRIC = "val_f1"
EARLY_STOP_PATIENCE = 8
GRAD_CLIP = 1.0

# Cache all samples in RAM after the first DB pass.
# Eliminates DB queries for epochs 2+.  Memory cost:
#   seismic/accel: ~300 MB – 1 GB
#   audio: ~25 GB (disable if RAM is limited)
CACHE_SAMPLES = True

# Quick sweep mode: set via SWEEP=1 env var to override EPOCHS
# for fast architecture comparison before full training runs.
_sweep = os.environ.get("SWEEP")
if _sweep:
    EPOCHS = int(_sweep) if _sweep.isdigit() else 5

BLOCK_SIZE = 60
USABLE_SIZE = 45
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

SAMPLE_SECONDS = 1

# =====================================================================
# 8. SPECTRAL PARAMETERS (Sensor-Adaptive)
# =====================================================================
#
# Audio at 16kHz produces rich spectrograms with many time frames.
# Seismic/accel at 100-200Hz produce tiny spectrograms — we use
# fewer mel bins and smaller FFT windows accordingly.

_SPECTRAL_PROFILES = {
    "audio": {
        "MEL_BINS": 64,
        "N_FFT": 1024,
        "HOP_LENGTH": 256,
    },
    "seismic": {
        "MEL_BINS": 32,
    },
    "accel": {
        "MEL_BINS": 32,
    },
}

_spectral = _SPECTRAL_PROFILES.get(TRAIN_SENSOR, {})
MEL_BINS = _spectral.get("MEL_BINS", 64)

_signal_length = int(REF_SAMPLE_RATE * SAMPLE_SECONDS)

# For audio, use the profile values. For low-rate sensors, compute dynamically.
if "N_FFT" in _spectral:
    N_FFT = _spectral["N_FFT"]
    HOP_LENGTH = _spectral["HOP_LENGTH"]
else:
    N_FFT = min(1024, 2 ** int(math.log2(_signal_length)))
    HOP_LENGTH = max(1, N_FFT // 4)

MEL_TOP_DB = 80
NOISE_KERNEL_SIZE = 51
BATCH_MODE = True

# =====================================================================
# 9. DATA AUGMENTATION & SYNTHESIS
# =====================================================================

SYNTHESIZE_BACKGROUND = True
SYNTHESIZE_PROBABILITY = 0.5
AUGMENT_SNR = True
AUGMENT_SNR_RANGE = (10, 30)
OVERSAMPLE_BACKGROUNDS = True

# =====================================================================
# 10. SENSOR-SPECIFIC MODEL HYPERPARAMETERS
# =====================================================================
#
# Each (MODEL_NAME, SENSOR) pair maps to a dict of hyperparameters.
# This replaces the old flat if/elif blocks — different sensors get
# architectures tuned to their signal length and frequency content.
#
# Key design rationale:
#   Audio (16kHz, 16000 samples) — large kernels/strides OK, deep nets,
#       rich mel spectrograms with MaxPool.
#   Seismic (200Hz, 200 samples) — small kernels, moderate depth,
#       tiny spectrograms with no MaxPool (AdaptiveAvgPool handles it).
#   Accel (200Hz, 200 samples, 3ch) — same as seismic structurally;
#       the 3 input channels provide cross-axis information naturally.

_HYPERPARAMS = {

    # -----------------------------------------------------------------
    # DetectionCNN (2D spectrogram)
    # -----------------------------------------------------------------
    ("DetectionCNN", "audio"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [5, 3],
        "STRIDES": [2, 1],
        "PADS": [2, 1],
        "POOL": True,
        "HIDDEN": 64,
        "DROPOUT": 0.2,
    },
    ("DetectionCNN", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [3, 3],
        "STRIDES": [1, 1],
        "PADS": [1, 1],
        "POOL": False,
        "HIDDEN": 64,
        "DROPOUT": 0.2,
    },
    ("DetectionCNN", "accel"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [3, 3],
        "STRIDES": [1, 1],
        "PADS": [1, 1],
        "POOL": False,
        "HIDDEN": 64,
        "DROPOUT": 0.2,
    },

    # -----------------------------------------------------------------
    # ClassificationCNN (2D spectrogram)
    # -----------------------------------------------------------------
    ("ClassificationCNN", "audio"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [32, 64, 128, 256],
        "KERNELS": [3, 3, 3, 3],
        "STRIDES": [1, 1, 1, 1],
        "PADS": [1, 1, 1, 1],
        "POOL": True,
        "HIDDEN": 512,
        "DROPOUT": 0.3,
    },
    ("ClassificationCNN", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [32, 64],
        "KERNELS": [3, 3],
        "STRIDES": [1, 1],
        "PADS": [1, 1],
        "POOL": False,
        "HIDDEN": 128,
        "DROPOUT": 0.3,
    },
    ("ClassificationCNN", "accel"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [32, 64],
        "KERNELS": [3, 3],
        "STRIDES": [1, 1],
        "PADS": [1, 1],
        "POOL": False,
        "HIDDEN": 128,
        "DROPOUT": 0.3,
    },

    # -----------------------------------------------------------------
    # WaveformClassificationCNN (1D raw waveform)
    # -----------------------------------------------------------------
    ("WaveformClassificationCNN", "audio"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [32, 64, 128],
        "KERNELS": [64, 32, 16],
        "STRIDES": [8, 4, 2],
        "HIDDEN": 256,
        "DROPOUT": 0.3,
    },
    ("WaveformClassificationCNN", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [32, 64, 128],
        "KERNELS": [7, 5, 3],
        "STRIDES": [2, 2, 1],
        "HIDDEN": 128,
        "DROPOUT": 0.3,
    },
    ("WaveformClassificationCNN", "accel"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [32, 64, 128],
        "KERNELS": [7, 5, 3],
        "STRIDES": [2, 2, 1],
        "HIDDEN": 128,
        "DROPOUT": 0.3,
    },

    # -----------------------------------------------------------------
    # ClassificationLSTM (CNN frontend → LSTM)
    # -----------------------------------------------------------------
    ("ClassificationLSTM", "audio"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [32, 16],
        "STRIDES": [8, 4],
        "POOLS": [4, 2],
        "HIDDEN": 128,
        "LAYERS": 3,
        "DIM": 64,
        "DROPOUT": 0.3,
    },
    ("ClassificationLSTM", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [7, 5],
        "STRIDES": [2, 2],
        "POOLS": [2, 2],
        "HIDDEN": 64,
        "LAYERS": 2,
        "DIM": 32,
        "DROPOUT": 0.2,
    },
    ("ClassificationLSTM", "accel"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [7, 5],
        "STRIDES": [2, 2],
        "POOLS": [2, 2],
        "HIDDEN": 64,
        "LAYERS": 2,
        "DIM": 32,
        "DROPOUT": 0.2,
    },

    # -----------------------------------------------------------------
    # BiGRU (CNN frontend + bidirectional GRU)
    # -----------------------------------------------------------------
    ("BiGRU", "audio"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [32, 16],
        "STRIDES": [8, 4],
        "POOLS": [4, 2],
        "HIDDEN": 128,
        "LAYERS": 2,
        "DIM": 64,
        "DROPOUT": 0.3,
    },
    ("BiGRU", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [7, 5],
        "STRIDES": [2, 2],
        "POOLS": [2, 2],
        "HIDDEN": 64,
        "LAYERS": 2,
        "DIM": 32,
        "DROPOUT": 0.2,
    },
    ("BiGRU", "accel"): {
        "LEARNING_RATE": 1e-3,
        "CHANNELS": [16, 32],
        "KERNELS": [7, 5],
        "STRIDES": [2, 2],
        "POOLS": [2, 2],
        "HIDDEN": 64,
        "LAYERS": 2,
        "DIM": 32,
        "DROPOUT": 0.2,
    },

    # -----------------------------------------------------------------
    # TCN (dilated causal convolutions)
    # -----------------------------------------------------------------
    ("TCN", "audio"): {
        "LEARNING_RATE": 1e-3,
        "TCN_CHANNELS": 64,
        "TCN_KERNEL_SIZE": 7,
        "TCN_LEVELS": 4,
        "HIDDEN": 128,
        "DROPOUT": 0.2,
    },
    ("TCN", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "TCN_CHANNELS": 64,
        "TCN_KERNEL_SIZE": 7,
        "TCN_LEVELS": 4,
        "HIDDEN": 128,
        "DROPOUT": 0.2,
    },
    ("TCN", "accel"): {
        "LEARNING_RATE": 1e-3,
        "TCN_CHANNELS": 64,
        "TCN_KERNEL_SIZE": 7,
        "TCN_LEVELS": 4,
        "HIDDEN": 128,
        "DROPOUT": 0.2,
    },

    # -----------------------------------------------------------------
    # InceptionTime (multi-scale parallel convolutions)
    # -----------------------------------------------------------------
    ("InceptionTime", "audio"): {
        "LEARNING_RATE": 1e-3,
        "NB_FILTERS": 64,
        "INCEPTION_KERNELS": [9, 19, 39],
        "INCEPTION_BLOCKS": 3,
        "HIDDEN": 256,
        "DROPOUT": 0.3,
        "INCEPTION_STEM_STRIDE": max(
            1, int(REF_SAMPLE_RATE * SAMPLE_SECONDS) // 200
        ),
    },
    ("InceptionTime", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "NB_FILTERS": 64,
        "INCEPTION_KERNELS": [9, 19, 39],
        "INCEPTION_BLOCKS": 3,
        "HIDDEN": 256,
        "DROPOUT": 0.3,
        "INCEPTION_STEM_STRIDE": 1,
    },
    ("InceptionTime", "accel"): {
        "LEARNING_RATE": 1e-3,
        "NB_FILTERS": 64,
        "INCEPTION_KERNELS": [9, 19, 39],
        "INCEPTION_BLOCKS": 3,
        "HIDDEN": 256,
        "DROPOUT": 0.3,
        "INCEPTION_STEM_STRIDE": 1,
    },

    # -----------------------------------------------------------------
    # IterativeMiniRocket (all sensors identical)
    # -----------------------------------------------------------------
    ("IterativeMiniRocket", "audio"): {
        "LEARNING_RATE": 1e-3,
        "DROPOUT": 0.3,
        "MINIROCKET_FEATURES": 1000,
    },
    ("IterativeMiniRocket", "seismic"): {
        "LEARNING_RATE": 1e-3,
        "DROPOUT": 0.3,
        "MINIROCKET_FEATURES": 1000,
    },
    ("IterativeMiniRocket", "accel"): {
        "LEARNING_RATE": 1e-3,
        "DROPOUT": 0.3,
        "MINIROCKET_FEATURES": 1000,
    },
}

# Apply the sensor-specific hyperparameters to the global namespace
_key = (MODEL_NAME, TRAIN_SENSOR)
if _key in _HYPERPARAMS:
    globals().update(_HYPERPARAMS[_key])
else:
    raise ValueError(
        f"No hyperparameter profile for ({MODEL_NAME}, {TRAIN_SENSOR}). "
        f"Available: {sorted(_HYPERPARAMS.keys())}"
    )

# =====================================================================
# 11. ROUTING LOGIC
# =====================================================================

SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "IterativeMiniRocket": "1D",
    "InceptionTime": "1D",
    "TCN": "1D",
    "BiGRU": "1D",
}
USE_MEL = SHAPE_MAP.get(MODEL_NAME, "1D") == "2D"

# =====================================================================
# 12. EXPERIMENT SNAPSHOT
# =====================================================================


def save_config_snapshot():
    """Dump all uppercase config variables to JSON in the run directory."""
    os.makedirs(RUN_DIR, exist_ok=True)

    config_dict = {}
    for key, value in list(globals().items()):
        if not key.isupper() or key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()
        elif isinstance(value, torch.device):
            config_dict[key] = str(value)
        elif isinstance(
            value, (int, float, str, list, dict, bool, tuple, type(None))
        ):
            config_dict[key] = value

    with open(JSON_LOG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)


# =====================================================================
# 13. ENSEMBLE CONFIGURATION
# =====================================================================

_ensemble_models_env = os.environ.get("ENSEMBLE_MODELS", "")
ENSEMBLE_MODELS = (
    [m for m in _ensemble_models_env.split(",") if m]
    if _ensemble_models_env
    else list(SHAPE_MAP.keys())
)
ENSEMBLE_WEIGHT_METRIC = "val_f1"
ENSEMBLE_WEIGHT_SCHEME = "linear"   # "linear" or "softmax"

ENSEMBLE_RUN_ID = os.environ.get(
    "ENSEMBLE_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S")
)
ENSEMBLE_DIR = os.path.join(
    "saved_models", "ensemble", TRAINING_MODE, ENSEMBLE_RUN_ID
)
ENSEMBLE_REPORT_PATH = os.path.join(ENSEMBLE_DIR, "ensemble_report.txt")
ENSEMBLE_WEIGHTS_PATH = os.path.join(
    "saved_models", "ensemble", TRAINING_MODE, "ensemble_weights.json"
)
