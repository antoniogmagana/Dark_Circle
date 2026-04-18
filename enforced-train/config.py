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
EPOCHS = 100
NUM_WORKERS = 96
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
# 2. DATA SOURCE
# =====================================================================
# Root of the pre-split parquet tree (relative to enforced-train/ working dir).
# Override with DATA_DIR env var when running from a different location.
DATA_DIR = os.environ.get("DATA_DIR", "../data_files/parsed")
DATA_SCAN_DIRS = ["train", "val", "test_iobt"]

# =====================================================================
# 3. TRAINING MODE
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
# Checkpoint and early-stopping metric is controlled by CHECKPOINT_METRIC (section 7c).
# Options: "mcc" | "val_f1" | "val_loss" | "val_acc" | "val_precision" | "val_recall"
# Note: val_f1 is always logged to meta.pt for evaluate_best_ensemble() model selection.

# =====================================================================
# 4. DATASET, SENSOR & CLASS CONSTANTS
# =====================================================================
# "iobt" "focal" "m3nvc"
TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]

# "audio" "seismic" "accel" — one sensor per training run
TRAIN_SENSOR = os.environ.get("TRAIN_SENSOR", "audio")
TRAIN_SENSORS = [TRAIN_SENSOR]

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

# Bit depth per sensor type (hardware spec)
BIT_DEPTH_MAP = {"audio": 16, "seismic": 24, "accel": 24}

# ADC max count per sensor (signed: 2^(bits-1))
ADC_SCALE_MAP = {s: 2 ** (b - 1) for s, b in BIT_DEPTH_MAP.items()}

# Per-channel ADC scales ordered to match channel concatenation in dataset.py
# audio=1ch, seismic=1ch, accel=3ch — channels stacked in TRAIN_SENSORS order
_SENSOR_CHANNELS = {"audio": 1, "seismic": 1, "accel": 3}
CHANNEL_ADC_SCALES = [
    ADC_SCALE_MAP[s] for s in TRAIN_SENSORS for _ in range(_SENSOR_CHANNELS[s])
]

# Semantic category names (used for category-level classification)
# if background used, always set to 0: "background"
CLASS_MAP = {0: "pedestrian", 1: "light", 2: "sport", 3: "utility"}

# Instance → category mapping (authoritative)
DATASET_VEHICLE_MAP = {
    "iobt": {
        "polaris0150pm": ["light", "polaris", "train"],
        "polaris0215pm": ["light", "polaris", "val"],
        "polaris0235pm_nolineofsig": ["light", "polaris", "test"],
        "warhog1135am": ["light", "warhog", "train"],
        "warhog1149am": ["light", "warhog", "val"],
        "warhog_nolineofsight": ["light", "warhog", "test"],
        "silverado0255pm": ["utility", "pickup", "train"],
        "silverado0315pm": ["utility", "pickup", "split"],
    },
    "focal": {
        "walk": ["pedestrian", "walk", "train"],
        "walk2": ["pedestrian", "walk", "split"],
        "bicycle": ["pedestrian", "bicycle", "train"],
        "bicycle2": ["pedestrian", "bicycle", "split"],
        "motor": ["light", "motorcycle", "train"],
        "motor2": ["light", "motorcycle", "split"],
        "scooter": ["light", "scooter", "train"],
        "scooter2": ["light", "scooter", "split"],
        "forester": ["utility", "forester", "train"],
        "forester2": ["utility", "forester", "split"],
        "mustang": ["sport", "mustang", "train"],
        "mustang0528": ["sport", "mustang", "val"],
        "mustang2": ["sport", "mustang", "test"],
        "pickup": ["utility", "pickup", "train"],
        "pickup2": ["utility", "pickup", "split"],
        "tesla": ["sport", "ev", "train"],
        "tesla2": ["sport", "ev", "split"],
    },
    "m3nvc": {
        "background": ["background", "background", "run"],
        "cx30": ["utility", "cx30", "run"],
        "miata": ["sport", "miata", "run"],
        "mustang": ["sport", "mustang", "run"],
        # "cx30_miata": 4,
        # "cx30_mustang": 4,
        # "miata_mustang": "sport",
        "gle350": ["utility", "gle350", "run"],
    },
}

# =====================================================================
# 5. DYNAMIC LABEL SPACE CONSTRUCTION
# =====================================================================

# Collect all instances across all datasets
ALL_INSTANCES = []
for ds_map in DATASET_VEHICLE_MAP.values():
    ALL_INSTANCES.extend(v[1] for v in ds_map.values())
ALL_INSTANCES = sorted(set(ALL_INSTANCES))

# Build reproducible instance-level class IDs
random.seed(INSTANCE_SEED)
shuffled_instances = ALL_INSTANCES.copy()
random.shuffle(shuffled_instances)

INSTANCE_TO_CLASS = {name: idx for idx, name in enumerate(shuffled_instances)}

# Optional override: set to a non-empty list to skip dynamic computation.
# Leave empty ([]) to compute inverse-frequency weights from the training set.
CLASS_WEIGHTS = []

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
# 6. MODEL SELECTION & DYNAMIC VERSIONING
# =====================================================================

MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    MODEL_NAME = input("Enter Model Name: ")

# 1. Generate or Retrieve RUN_ID
# If evaluating, pass RUN_ID="20260308_2032".
# Otherwise a new timestamp is generated.
RUN_ID = os.environ.get("RUN_ID")
if not RUN_ID:
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. Build the nested directory structure (includes sensor for per-sensor runs)
RUN_DIR = os.path.join("saved_models", TRAINING_MODE, TRAIN_SENSOR, MODEL_NAME, RUN_ID)
CACHE_DIR = os.path.join("saved_models", "cache")

# 3. Define the specific file paths inside that new folder
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
META_SAVE_PATH = os.path.join(RUN_DIR, "meta.pt")
IMG_SAVE_PATH = os.path.join(RUN_DIR, "conf_matrix.png")
JSON_LOG_PATH = os.path.join(RUN_DIR, "hyperparameters.json")
METRICS_LOG_PATH = os.path.join(RUN_DIR, "metrics.csv")

TRAIN_STEPS_PER_EPOCH = 400
VAL_STEPS_PER_EPOCH = 128

# Time-scale knobs
SAMPLE_SECONDS = 2
WINDOW_STEP = 0.1  # sliding window stride in seconds

# Mel spectrogram parameters
MEL_BINS = 64
MEL_HOP_LENGTH = 512
MEL_TOP_DB = 80
NOISE_KERNEL_SIZE = 51

BATCH_MODE = True

# =====================================================================
# 7. DATA AUGMENTATION
# =====================================================================
# Toggle to dynamically add noise to the raw waveform during training
AUGMENT_SNR = True

# The minimum and maximum SNR (in decibels) to apply when augmenting
# Sensor-specific SNR ranges
AUGMENT_SNR_RANGE_AUDIO    = (10, 30)   # appropriate for 16kHz high-energy signal
AUGMENT_SNR_RANGE_SEISMIC  = (20, 40)   # less aggressive for low-SNR 200Hz signal
AUGMENT_SNR_RANGE = AUGMENT_SNR_RANGE_AUDIO if TRAIN_SENSOR == "audio" else AUGMENT_SNR_RANGE_SEISMIC

# =====================================================================
# 7b. CLASS BALANCE
# =====================================================================
CLASS_WEIGHT_CAP          = 10.0   # inverse-frequency weight ceiling (tunable)
USE_WEIGHTED_SAMPLER      = True   # WeightedRandomSampler for train DataLoader
WEIGHTED_SAMPLER_STRENGTH = 1.0    # 1.0=fully balanced, 0.0=uniform

# =====================================================================
# 7c. TRAINING PROCEDURE
# =====================================================================
EARLY_STOP_PATIENCE = 15           # was hardcoded 8 in train.py
CHECKPOINT_METRIC   = "mcc"        # "mcc" | "val_f1" | "val_loss" | "val_acc"

# =====================================================================
# 7d. HARDWARE & PRECISION
# =====================================================================
PREFETCH_FACTOR    = 16
PIN_MEMORY         = True
PERSISTENT_WORKERS = True
TORCH_COMPILE      = True
AMP_ENABLED        = True
AMP_DTYPE          = "bfloat16"    # H100 native — no loss scaling needed

# =====================================================================
# 8. MODEL HYPERPARAMETERS & CONTROL FLOW
# =====================================================================
BASE_LR = 1e-3
BASE_DROPOUT = 0.3

# Sequence length derived from the active sensor/dataset combination.
# All per-model kernel/stride values below are computed from this so they
# remain valid regardless of which sensor or sample rate is selected.
SEQ_LEN = int(REF_SAMPLE_RATE * SAMPLE_SECONDS)

# Explicit mel parameters — global for both sensors.
# Tuned for seismic (200Hz × 2s = 400 samples → ~18 mel frames at HOP=16).
# Audio (16kHz × 2s = 32000 samples) produces ~1875 frames — no ceiling concern.
# N_FFT=128 gives 65 frequency bins, satisfying MEL_BINS=64 without degenerate filterbank.
N_FFT = 128
HOP_LENGTH = 16

# --- Detection CNN ---
if MODEL_NAME == "DetectionCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    KERNELS = [5, 3]
    STRIDES = [2, 1]
    PADS = [2, 1]
    HIDDEN = 64

# --- Classification CNN ---
elif MODEL_NAME == "ClassificationCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [32, 64, 128, 256]
    KERNEL = 3
    PADS = 1
    HIDDEN = 512
    DROPOUT = 0.3

# --- Waveform 1D CNN ---
elif MODEL_NAME == "WaveformClassificationCNN":
    LEARNING_RATE = 1e-3
    CHANNELS = [32, 64, 128]
    HIDDEN = 256
    DROPOUT = 0.3
    # Kernels and strides derived from SEQ_LEN so the 3-layer frontend
    # always produces a valid flat feature vector regardless of sample rate.
    # Layer 1 targets ~50 samples; layer 2 ~20; layer 3 ~10.
    _s0 = max(4, SEQ_LEN // 50)
    _k0 = max(8, _s0 * 2)
    _len1 = (SEQ_LEN - _k0) // _s0 + 1
    _s1 = max(2, _len1 // 20)
    _k1 = max(4, _s1 * 2)
    _len2 = (_len1 - _k1) // _s1 + 1
    _s2 = max(1, _len2 // 10)
    _k2 = max(4, _s2 * 2)
    KERNELS = [_k0, _k1, _k2]
    STRIDES = [_s0, _s1, _s2]

# --- LSTM Networks ---
elif MODEL_NAME == "ClassificationLSTM":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    HIDDEN = 128
    LAYERS = 3
    DIM = 64
    DROPOUT = 0.3
    POOLS = [2, 2]
    # CNN frontend sized for SEQ_LEN: layer 1 targets ~100 samples,
    # after pool ~50; layer 2 targets ~25 RNN timesteps after pool.
    _s0 = max(2, SEQ_LEN // 100)
    _k0 = max(8, _s0 * 2)
    _len1 = ((SEQ_LEN - _k0) // _s0 + 1) // POOLS[0]
    _s1 = max(1, _len1 // 25)
    _k1 = max(4, _s1 * 2)
    KERNELS = [_k0, _k1]
    STRIDES = [_s0, _s1]

# --- InceptionTime ---
elif MODEL_NAME == "InceptionTime":
    LEARNING_RATE = 1e-3
    NB_FILTERS = 64  # output channels per parallel branch
    INCEPTION_KERNELS = [
        9,
        19,
        39,
    ]  # kernel sizes (9≈45ms, 19≈95ms, 39≈195ms at 200Hz post-stem)
    INCEPTION_BLOCKS = 3  # inception modules; residual shortcut every 3
    HIDDEN = 256
    DROPOUT = 0.3
    # Stem stride normalises T to ~200 samples before the inception blocks.
    # At seismic-only rates (SEQ_LEN≈200), stride=1 → Identity (no change).
    # At audio rates (SEQ_LEN=16000), stride=80 -> 200 post-stem samples,
    # keeping INCEPTION_KERNELS meaningful and tensors small (B=128).
    INCEPTION_STEM_STRIDE = max(1, SEQ_LEN // 200)

# --- BiGRU ---
elif MODEL_NAME == "BiGRU":
    LEARNING_RATE = 1e-3
    CHANNELS = [16, 32]
    HIDDEN = 128
    LAYERS = 2
    DIM = 64
    DROPOUT = 0.3
    POOLS = [2, 2]
    # Same frontend sizing logic as ClassificationLSTM.
    _s0 = max(2, SEQ_LEN // 100)
    _k0 = max(8, _s0 * 2)
    _len1 = ((SEQ_LEN - _k0) // _s0 + 1) // POOLS[0]
    _s1 = max(1, _len1 // 25)
    _k1 = max(4, _s1 * 2)
    KERNELS = [_k0, _k1]
    STRIDES = [_s0, _s1]

# =====================================================================
# 9. ROUTING LOGIC (Replacing Circular Dependencies)
# =====================================================================


# Map the current model to its required input shape
SHAPE_MAP = {
    "DetectionCNN": "2D",
    "ClassificationCNN": "2D",
    "WaveformClassificationCNN": "1D",
    "ClassificationLSTM": "1D",
    "InceptionTime": "1D",
    "BiGRU": "1D",
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

            # Handle NumPy arrays which are not JSON-serializable
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
