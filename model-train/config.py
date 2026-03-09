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
BATCH_SIZE = 128
EPOCHS = 10
NUM_WORKERS = 32
LOG_INTERVAL = 10

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
    "password": os.environ.get("DB_PASSWORD", "default_password"),
    "host": "localhost",
    "port": 5432,
}

# =====================================================================
# 3. TRAINING MODE (NEW)
# =====================================================================
# Options:
#   "detection"  -> binary: background vs vehicle
#   "category"   -> multi-class: background/light/heavy (CLASS_MAP)
#   "instance"   -> each vehicle instance is its own class
TRAINING_MODE = os.environ.get("TRAINING_MODE", "detection")

# Reproducible instance-level class IDs
INSTANCE_SEED = 0

# =====================================================================
# 4. DATASET, SENSOR & CLASS CONSTANTS
# =====================================================================
TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]
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
CLASS_MAP = {0: "background", 1: "light", 2: "heavy"}

# Instance → category mapping (authoritative)
DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm": 1,
        "polaris0215pm": 1,
        "polaris0235pm_nolineofsig": 1,
        "warhog1135am": 1,
        "warhog1149am": 1,
        "warhog_nolineofsight": 1,
        "silverado0255pm": 2,
        "silverado0315pm": 2,
    },
    "focal": {
        "walk": 0,
        "walk2": 0,
        "bicycle": 1,
        "bicycle2": 1,
        "motor": 1,
        "motor2": 1,
        "scooter": 1,
        "scooter2": 1,
        "forester": 2,
        "forester2": 2,
        "mustang": 2,
        "mustang0528": 2,
        "mustang2": 2,
        "pickup": 2,
        "pickup2": 2,
        "tesla": 2,
        "tesla2": 2,
    },
    "m3nvc": {
        "background": 0,
        "cx30": 2,
        "miata": 2,
        "mustang": 2,
        "cx30_miata": 2,
        "cx30_mustang": 2,
        "miata_mustang": 2,
        "gle350": 2,
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
    CLASS_WEIGHTS = [1.0, 1.0, 1.0]
elif TRAINING_MODE == "instance":
    NUM_CLASSES = len(INSTANCE_TO_CLASS)
else:
    raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")

# =====================================================================
# 6. MODEL SELECTION & DYNAMIC VERSIONING
# =====================================================================

MODEL_NAME = os.environ.get("MODEL_NAME", "DetectionCNN")

# 1. Generate or Retrieve RUN_ID
# If evaluating, we can pass RUN_ID="20260308_2032". Otherwise, it generates a new timestamp.
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

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
CHUNK_SECONDS = 15

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
DET_CNN_LR = 1e-3
DET_CNN_CHANNELS = [16, 32]
DET_CNN_KERNELS = [5, 3]
DET_CNN_STRIDES = [2, 1]
DET_CNN_PADS = [2, 1]
DET_CNN_HIDDEN = 64

# --- Classification CNN ---
CLASS_CNN_LR = 5e-4
CLASS_CNN_CHANNELS = [32, 64, 128, 256]
CLASS_CNN_KERNEL = 3
CLASS_CNN_PAD = 1
CLASS_CNN_HIDDEN = 512
CLASS_CNN_DROPOUT = 0.4

# --- Waveform 1D CNN ---
WAVE_CNN_LR = 1e-3
WAVE_CNN_CHANNELS = [32, 64, 128]
WAVE_CNN_KERNELS = [64, 32, 16]
WAVE_CNN_STRIDES = [8, 4, 2]
WAVE_CNN_HIDDEN = 256

# --- LSTM Networks ---
LSTM_LR = 1e-3
LSTM_CNN_CHANNELS = [16, 32]
LSTM_CNN_KERNELS = [32, 16]
LSTM_CNN_STRIDES = [8, 4]
LSTM_CNN_POOLS = [4, 2]
LSTM_HIDDEN = 128
LSTM_LAYERS = 3
LSTM_FC_DIM = 64
LSTM_DROPOUT = BASE_DROPOUT

# --- miniROCKET ---
ROCKET_NUM_KERNELS = 10000
ROCKET_ALPHAS = np.logspace(-3, 3, 10)
ROCKET_MAX_SAMPLES = 50000  
ROCKET_CV_FOLDS = 5         

# =====================================================================
# 9. ROUTING LOGIC (Replacing Circular Dependencies)
# =====================================================================

# Map the current model to its specific learning rate
LR_MAP = {
    "DetectionCNN": DET_CNN_LR,
    "ClassificationCNN": CLASS_CNN_LR,
    "WaveformClassificationCNN": WAVE_CNN_LR,
    "ClassificationLSTM": LSTM_LR,
    "ClassificationMiniRocket": None,  # Non-gradient
}
LEARNING_RATE = LR_MAP.get(MODEL_NAME, BASE_LR)

# Map the current model to its required input shape
SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "ClassificationMiniRocket": "1D",
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


# ===========================================================
# Bash script for looping over training pipeline
# ===========================================================

# copy this into terminal to loop over all models in one go!
"""
for mode in detection category instance; do
    for model in DetectionCNN ClassificationCNN WaveformClassificationCNN ClassificationLSTM; do
        # 1. Generate a unique timestamp for this specific run
        CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
        
        echo "============================================================"
        echo "STARTING RUN -> MODE: $mode | MODEL: $model | RUN_ID: $CURRENT_RUN_ID"
        echo "============================================================"
        
        # 2. Pass the RUN_ID to the training script
        RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode MODEL_NAME=$model poetry run python train.py
        
        # 3. Pass the EXACT SAME RUN_ID to the eval script so it finds the right folder
        if [ $? -eq 0 ]; then
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode MODEL_NAME=$model poetry run python eval.py
        else
            echo "Training failed for $model in $mode mode. Skipping eval."
        fi
        
    done
    
    # 4. Handle the MiniRocket run with its own timestamp
    ROCKET_RUN_ID=$(date +%Y%m%d_%H%M%S)
    echo "============================================================"
    echo "STARTING MINIROCKET RUN -> MODE: $mode | RUN_ID: $ROCKET_RUN_ID"
    echo "============================================================"
    RUN_ID=$ROCKET_RUN_ID TRAINING_MODE=$mode poetry run python train_rocket.py
done
"""

# use this syntax in bash to evaluate a single historical model
"""
RUN_ID="20260308_203200" TRAINING_MODE="detection" MODEL_NAME="DetectionCNN" poetry run python eval.py
"""
