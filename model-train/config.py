import os
import random
import torch


# ===========================================================
# Globals
# ===========================================================


# ============================================================
# Training hyperparameters
# ============================================================

# Core training loop settings
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 5
NUM_WORKERS = 32
LOG_INTERVAL = 50

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
TRAINING_MODE = "category"

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
    for inst_list in ds_map.values():
        ALL_INSTANCES.extend(inst_list)

ALL_INSTANCES = sorted(set(ALL_INSTANCES))

# Build reproducible instance-level class IDs
random.seed(INSTANCE_SEED)
shuffled_instances = ALL_INSTANCES.copy()
random.shuffle(shuffled_instances)

INSTANCE_TO_CLASS = {name: idx for idx, name in enumerate(shuffled_instances)}

# Determine number of classes based on training mode
if TRAINING_MODE == "detection":
    NUM_CLASSES = 2

elif TRAINING_MODE == "category":
    NUM_CLASSES = len(CLASS_MAP)

elif TRAINING_MODE == "instance":
    NUM_CLASSES = len(INSTANCE_TO_CLASS)

else:
    raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")

# =====================================================================
# 6. MODEL SELECTION
# =====================================================================

MODEL_NAME = "ClassificationCNN"
MODEL_SAVE_PATH = f"saved_models/{MODEL_NAME}_best.pth"

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

# Model input compatibility
WAVEFORM_ONLY_MODELS = [
    "WaveformCNN",
    "WaveformClassificationCNN",
    "RawNet",
    "RawNet2",
    "TCN",
    "WaveNet",
    "MiniRocket",
    "Rocket",
    "MultiRocket",
    "LSTM1D",
    "GRU1D",
]

MEL_ONLY_MODELS = [
    "MelCNN",
    "CRNN",
    "ResNetAudio",
    "VGGish",
    "AudioSpectrogramCNN",
    "MobileNetAudio",
]

EITHER_MODELS = [
    "ClassificationCNN",
    "DetectionCNN",
    "GenericCNN",
]

# ============================================================
# Mel spectrogram parameters
# ============================================================

# FFT window size
N_FFT = 1024

# Hop length between STFT frames
HOP_LENGTH = 256

# Number of mel bins
MEL_BINS = 64

USE_MEL = True

BATCH_MODE = True
