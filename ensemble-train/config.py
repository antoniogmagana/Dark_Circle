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
# 3. TRAINING MODE
# =====================================================================
#   "detection"  -> binary: background vs vehicle
#   "category"   -> multi-class by vehicle category
#   "instance"   -> each vehicle instance is its own class

TRAINING_MODE = os.environ.get("TRAINING_MODE")
if not TRAINING_MODE:
    TRAINING_MODE = input('Enter Training Mode ["detection", "category", "instance"]: ')

INSTANCE_SEED = 0

# =====================================================================
# 4. SENSOR SELECTION (Ensemble: one model per sensor)
# =====================================================================

ALL_SENSORS = ["audio", "seismic", "accel"]

TRAIN_SENSOR = os.environ.get("TRAIN_SENSOR")
if not TRAIN_SENSOR:
    TRAIN_SENSOR = input('Enter Sensor ["audio", "seismic", "accel"]: ')

# Backward-compatible list form (dataset.py iterates this)
TRAIN_SENSORS = [TRAIN_SENSOR]

# audio=1 channel, seismic=1 channel, accel=3 channels (x, y, z)
IN_CHANNELS = 3 if TRAIN_SENSOR == "accel" else 1

# =====================================================================
# 5. DATASET & CLASS CONSTANTS
# =====================================================================

TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]

ACOUSTIC_SR = 16000

# Native sample rates per dataset and sensor
NATIVE_SR = {
    "iobt":  {"audio": 16000, "seismic": 100, "accel": 100},
    "focal": {"audio": 16000, "seismic": 100, "accel": 100},
    "m3nvc": {"audio": 1600,  "seismic": 200, "accel": 200},
}

# Reference sample rate for the active sensor across all datasets
REF_SAMPLE_RATE = max(NATIVE_SR[ds][TRAIN_SENSOR] for ds in TRAIN_DATASETS)

# Semantic category names
CLASS_MAP = {0: "pedestrian", 1: "light", 2: "sport", 3: "utility"}

# Instance → category mapping (authoritative, per-dataset)
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
# 6. DYNAMIC LABEL SPACE
# =====================================================================

ALL_INSTANCES = sorted({
    name
    for ds_map in DATASET_VEHICLE_MAP.values()
    for name in ds_map.keys()
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
# 7. MODEL SELECTION & VERSIONING
# =====================================================================

MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME = input("Enter Model Name: ")

RUN_ID = os.environ.get("RUN_ID")
if not RUN_ID:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Directory layout: saved_models/{mode}/{sensor}/{model}/{run_id}/
RUN_DIR = os.path.join("saved_models", TRAINING_MODE, TRAIN_SENSOR, MODEL_NAME, RUN_ID)
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
META_SAVE_PATH = os.path.join(RUN_DIR, "meta.pt")
IMG_SAVE_PATH = os.path.join(RUN_DIR, "conf_matrix.png")
JSON_LOG_PATH = os.path.join(RUN_DIR, "hyperparameters.json")
METRICS_LOG_PATH = os.path.join(RUN_DIR, "metrics.csv")

# =====================================================================
# 8. CORE TRAINING HYPERPARAMETERS
# =====================================================================

BATCH_SIZE = 128
EPOCHS = 30
NUM_WORKERS = 32
LOG_INTERVAL = 10

BEST_MODEL_METRIC = "val_f1"
EARLY_STOP_PATIENCE = 8
GRAD_CLIP = 1.0

BLOCK_SIZE = 60
USABLE_SIZE = 45
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

SAMPLE_SECONDS = 1

# =====================================================================
# 9. SPECTRAL PARAMETERS (Dynamic — adapts to sensor sample rate)
# =====================================================================

_signal_length = int(REF_SAMPLE_RATE * SAMPLE_SECONDS)

N_FFT = min(1024, 2 ** int(math.log2(_signal_length)))
HOP_LENGTH = max(1, N_FFT // 4)

MEL_BINS = 64
MEL_HOP_LENGTH = 512
MEL_TOP_DB = 80
NOISE_KERNEL_SIZE = 51

BATCH_MODE = True

# =====================================================================
# 10. DATA AUGMENTATION & SYNTHESIS
# =====================================================================

SYNTHESIZE_BACKGROUND = True
SYNTHESIZE_PROBABILITY = 0.5
AUGMENT_SNR = True
AUGMENT_SNR_RANGE = (10, 30)
OVERSAMPLE_BACKGROUNDS = True

# =====================================================================
# 11. MODEL-SPECIFIC HYPERPARAMETERS
# =====================================================================

BASE_LR = 1e-3
BASE_DROPOUT = 0.3

if MODEL_NAME == "DetectionCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    KERNELS = [5, 3]
    STRIDES = [2, 1]
    PADS = [2, 1]
    HIDDEN = 64

elif MODEL_NAME == "ClassificationCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [32, 64, 128, 256]
    KERNEL = 3
    PADS = 1
    HIDDEN = 512
    DROPOUT = 0.3

elif MODEL_NAME == "WaveformClassificationCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [32, 64, 128]
    KERNELS = [64, 32, 16]
    STRIDES = [8, 4, 2]
    HIDDEN = 256
    DROPOUT = 0.3

elif MODEL_NAME == "ClassificationLSTM":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    KERNELS = [32, 16]
    STRIDES = [8, 4]
    POOLS = [4, 2]
    HIDDEN = 128
    LAYERS = 3
    DIM = 64
    DROPOUT = 0.3

elif MODEL_NAME == "IterativeMiniRocket":
    LEARNING_RATE = 1e-3
    DROPOUT = 0.3
    MINIROCKET_FEATURES = 1000

# =====================================================================
# 12. ROUTING LOGIC
# =====================================================================

SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "IterativeMiniRocket": "1D",
}
USE_MEL = SHAPE_MAP.get(MODEL_NAME, "1D") == "2D"

# =====================================================================
# 13. EXPERIMENT SNAPSHOT
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
        elif isinstance(value, (int, float, str, list, dict, bool, tuple, type(None))):
            config_dict[key] = value

    with open(JSON_LOG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)
