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
BATCH_SIZE = 128
EPOCHS = 5
NUM_WORKERS = 12
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
        1: [
            "polaris0150pm",
            "polaris0215pm",
            "polaris0235pm_nolineofsig",
            "warhog1135am",
            "warhog1149am",
            "warhog_nolineofsight",
        ],
        2: ["silverado0255pm", "silverado0315pm"],
    },
    "focal": {
        0: ["walk", "walk2"],
        1: ["bicycle", "bicycle2", "motor", "motor2", "scooter", "scooter2"],
        2: [
            "forester",
            "forester2",
            "mustang",
            "mustang0528",
            "mustang2",
            "pickup",
            "pickup2",
            "tesla",
            "tesla2",
        ],
    },
    "m3nvc": {
        0: ["background"],
        2: [
            "cx30",
            "miata",
            "mustang",
            "cx30_miata",
            "cx30_mustang",
            "miata_mustang",
            "gle350",
        ],
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

USE_MEL = True
BATCH_MODE = True
