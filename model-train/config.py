import os
import random
import torch
import json
from datetime import datetime
import numpy as np


# ===========================================================
# Globals
# ===========================================================

# ============================================================
# Training hyperparameters
# ============================================================

# Core training loop settings
BATCH_SIZE = 1024
EPOCHS = 5
NUM_WORKERS = 32
LOG_INTERVAL = 10
BLOCK_SIZE = 60
USABLE_SIZE = 45

# Checkpoint + evaluation output directories
CHECKPOINT_DIR = "./checkpoints"
EVAL_RESULTS_DIR = "./eval_results"

# How many batches eval.py should run before stopping
EVAL_STEPS = 200

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
# 3. TRAINING MODE (NEW)
# =====================================================================
# Options:
#   "detection"  -> binary: background vs vehicle
#   "category"   -> multi-class: background/light/heavy (CLASS_MAP)
#   "instance"   -> each vehicle instance is its own class
TRAINING_MODE = os.environ.get("TRAINING_MODE")
if not TRAINING_MODE:
    TRAINING_MODE = input('Enter Training Mode ["detection", "category", "instance"]: ')

# Reproducible instance-level class IDs
INSTANCE_SEED = 0

# =====================================================================
# MODEL SELECTION CRITERIA
# =====================================================================
# Options: "val_acc", "val_loss", "val_f1", "val_precision", "val_recall"
BEST_MODEL_METRIC = "val_f1"

# =====================================================================
# 4. DATASET, SENSOR & CLASS CONSTANTS
# =====================================================================
# "iobt" "focal" "m3nvc"
TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]

# "audio" "seismic" "accel"
TRAIN_SENSORS = ["audio", "seismic"]  # accel can be added later

# Derived: audio=1, seismic=1, accel=3
IN_CHANNELS = len(TRAIN_SENSORS) + (2 if "accel" in TRAIN_SENSORS else 0)

# Target sample rate all tensors will be upsampled to for the CNN
ACOUSTIC_SR = 16000

# Native sample rates per dataset and sensor
NATIVE_SR = {
    "iobt": {"audio": 16000, "seismic": 100, "accel": 100},
    "focal": {"audio": 16000, "seismic": 100, "accel": 100},
    "m3nvc": {"audio": 1600, "seismic": 200, "accel": 200},
}

# Global reference sample rate for all sensors/datasets
REF_SAMPLE_RATE = max(NATIVE_SR[ds][s] for ds in TRAIN_DATASETS for s in TRAIN_SENSORS)

# Semantic category names (used for category-level classification)
# if background used, always set to 0: "background"
CLASS_MAP = {0: "background", 1: "pedestrian", 2: "light", 3: "sport", 4: "utility"}

# Instance → category mapping (authoritative)
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
        # "cx30_miata": 4,
        # "cx30_mustang": 4,
        "miata_mustang": "sport",
        "gle350": "utility",
    },
}

# =====================================================================
# 5. DYNAMIC LABEL SPACE CONSTRUCTION (NEW)
# =====================================================================

# Collect all instances across all datasets
ALL_INSTANCES = []
for ds_map in DATASET_VEHICLE_MAP.values():
    ALL_INSTANCES.extend(ds_map.keys())

ALL_INSTANCES = sorted(set(ALL_INSTANCES))

# Build reproducible instance-level class IDs
random.seed(INSTANCE_SEED)
shuffled_instances = ALL_INSTANCES.copy()
random.shuffle(shuffled_instances)

INSTANCE_TO_CLASS = {name: idx for idx, name in enumerate(shuffled_instances)}

CLASS_WEIGHTS = []

# Determine number of classes based on training mode
if TRAINING_MODE == "detection":
    NUM_CLASSES = 2
    CLASS_WEIGHTS = [1.0, 1.0]
elif TRAINING_MODE == "category":
    NUM_CLASSES = len(CLASS_MAP)
    CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0] # based on classification classes
elif TRAINING_MODE == "instance":
    NUM_CLASSES = len(INSTANCE_TO_CLASS)
else:
    raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")

# =====================================================================
# 6. MODEL SELECTION & DYNAMIC VERSIONING
# =====================================================================

MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME = input('Enter Model Name: ')

# 1. Generate or Retrieve RUN_ID
# If evaluating, we can pass RUN_ID="20260308_2032". Otherwise, it generates a new timestamp.
RUN_ID = os.environ.get("RUN_ID")
if not RUN_ID:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. Build the nested directory structure
RUN_DIR = os.path.join("saved_models", TRAINING_MODE, MODEL_NAME, RUN_ID)

# 3. Define the specific file paths inside that new folder
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
META_SAVE_PATH = os.path.join(RUN_DIR, "meta.pt")
IMG_SAVE_PATH = os.path.join(RUN_DIR, "conf_matrix.png")
JSON_LOG_PATH = os.path.join(RUN_DIR, "hyperparameters.json")
METRICS_LOG_PATH = os.path.join(RUN_DIR, "metrics.csv")

BATCH_SIZE = 128
TRAIN_STEPS_PER_EPOCH = 50
VAL_STEPS_PER_EPOCH = 16

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

# Time-scale knobs
SAMPLE_SECONDS = 1

# Mel spectrogram parameters
MEL_BINS = 64
MEL_HOP_LENGTH = 512
MEL_TOP_DB = 80
NOISE_KERNEL_SIZE = 51

# FFT window size
N_FFT = 1024

# Hop length between STFT frames
HOP_LENGTH = 256

BATCH_MODE = True

# =====================================================================
# 7. DATA AUGMENTATION & SYNTHESIS
# =====================================================================
# Toggle to dynamically inject generated background noise during training
SYNTHESIZE_BACKGROUND = True

# The probability (0.0 to 1.0) of adding a SYNTHETIC background sample
SYNTHESIZE_PROBABILITY = 0.5

# Toggle to dynamically add noise to the raw waveform during training
AUGMENT_SNR = True

# The minimum and maximum SNR (in decibels) to apply when augmenting
AUGMENT_SNR_RANGE = (10, 30)

# add extra synthetic background samples
OVERSAMPLE_BACKGROUNDS = True

# =====================================================================
# 8. MODEL HYPERPARAMETERS & CONTROL FLOW
# =====================================================================
BASE_LR = 1e-3
BASE_DROPOUT = 0.3

# --- Detection CNN ---
if MODEL_NAME == "DetectionCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    KERNELS = [5, 3]
    STRIDES = [2, 1]
    PADS = [2, 1]
    HIDDEN = 64

# --- Classification CNN ---
if MODEL_NAME == "ClassificationCNN":
    LEARNING_RATE = 5e-4
    CHANNELS = [32, 64, 128, 256]
    KERNEL = 3
    PADS = 1
    HIDDEN = 512
    DROPOUT = 0.4

# --- Waveform 1D CNN ---
if MODEL_NAME == "WaveformClassificationCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [32, 64, 128]
    KERNELS = [64, 32, 16]
    STRIDES = [8, 4, 2]
    HIDDEN = 256
    DROPOUT = 0.3

# --- LSTM Networks ---
if MODEL_NAME == "ClassificationLSTM":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    KERNELS = [32, 16]
    STRIDES = [8, 4]
    POOLS = [4, 2]
    HIDDEN = 128
    LAYERS = 3
    DIM = 64
    DROPOUT = 0.3

# --- miniROCKET ---
if MODEL_NAME == "IterativeMiniRocket":
    LEARNING_RATE = 1e-3
    DROPOUT = 0.3
    MINIROCKET_FEATURES = 1000
    # The tsai extractor defaults to 10,000 kernels automatically 

# =====================================================================
# 9. ROUTING LOGIC (Replacing Circular Dependencies)
# =====================================================================


# Map the current model to its required input shape
SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "IterativeMiniRocket": "1D",
}
USE_MEL = SHAPE_MAP.get(MODEL_NAME, "1D") == "2D"

# =====================================================================
# 10. EXPERIMENT TRACKING UTILITY
# =====================================================================


def save_config_snapshot():
    """
    Scans the config namespace for hyperparameter variables and dumps them
    into a JSON file inside the specific run's directory.
    """
    os.makedirs(RUN_DIR, exist_ok=True)

    config_dict = {}

    # Iterate through all variables in this file
    for key, value in list(globals().items()):
        # Only grab standard uppercase configuration variables
        if key.isupper() and not key.startswith("_"):

            # Handle NumPy arrays (like ROCKET_ALPHAS) which JSON hates
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            # Handle PyTorch devices
            elif isinstance(value, torch.device):
                config_dict[key] = str(value)
            # Handle standard JSON-serializable types
            elif isinstance(
                value, (int, float, str, list, dict, bool, tuple, type(None))
            ):
                config_dict[key] = value

    with open(JSON_LOG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)