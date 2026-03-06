import os
import torch

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
# 3. DATASET, SENSOR & CLASS CONSTANTS
# =====================================================================
TRAIN_DATASETS = ["iobt", "focal"]
TRAIN_SENSORS = ["audio", "seismic"]
IN_CHANNELS = len(TRAIN_SENSORS)

ACOUSTIC_SR = 16000
SEISMIC_SR = 100

CLASS_MAP = {0: "background", 1: "vehicle"}
DATASET_WEIGHTS = {"iobt": 0.5, "focal": 0.5}

DATASET_VEHICLE_MAP = {
    "iobt": {1: ["polaris0150pm", "silverado0255pm"]},
    "focal": {1: ["bicycle", "bicycle2"]},
}

# =====================================================================
# 4. REPEATED SETTINGS & ORCHESTRATION CONTROLS
# =====================================================================
MODEL_SAVE_PATH = "saved_models/best_vehicle_model.pth"

# Training Loop Limits
BATCH_SIZE = 128
TRAIN_STEPS_PER_EPOCH = 50
VAL_STEPS_PER_EPOCH = 16
EVAL_STEPS = 50

# Temporal Split Logic
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

# Signal Processing & Preprocessing Controls
MEL_BINS = 64
MEL_HOP_LENGTH = 512
MEL_TOP_DB = 80
NOISE_KERNEL_SIZE = 51
