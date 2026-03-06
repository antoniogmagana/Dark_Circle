import os
import torch

# =====================================================================
# 1. HARDWARE CONFIGURATION
# =====================================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =====================================================================
# 2. DATABASE CONFIGURATION
# =====================================================================
DB_CONN_PARAMS = {
    "dbname": "lvc_db",
    "user": "lvc_toolkit",
    "password": os.environ.get("DB_PASSWORD", "default_password_if_any"),
    "host": "localhost",
    "port": 5432,
}

# Fetch size: Exactly 64 windows at 16kHz
DB_CHUNK_SIZE = 1024000

# =====================================================================
# 3. TRAINING SCOPE & TOGGLES
# =====================================================================
# Options: ["iobt"], ["focal"], or ["iobt", "focal"]
TRAIN_DATASETS = ["iobt", "focal"]

# Options: ["audio"], ["seismic"], ["accel"]
TRAIN_SENSORS = ["audio", "seismic", "accel"]

# Automatically calculates required input channels for the CNN
IN_CHANNELS = len(TRAIN_SENSORS)

# Dataset sampling weights to balance Focal (smaller) with IoBT (larger)
DATASET_WEIGHTS = {"iobt": 0.4, "focal": 0.6}

# 20% chance to drop an entire sensor modality to train robustness
SENSOR_DROPOUT_PROB = 0.2

# Weight serialization path
MODEL_NAME = f"model_{'_'.join(TRAIN_DATASETS)}_{'_'.join(TRAIN_SENSORS)}.pth"
MODEL_SAVE_PATH = os.path.join("../models", MODEL_NAME)

# =====================================================================
# 4. CATEGORICAL & DATASET MAPPING
# =====================================================================
BATCH_SIZE = 64
ACOUSTIC_SR = 16000
SEISMIC_SR = 100

# Global Class Mapping
CLASS_MAP = {0: "background", 1: "light", 2: "heavy"}

# String identifiers found in the PostgreSQL table names
DATASET_VEHICLE_MAP = {
    "focal": {
        1: ["bicycle", "bicycle2", "walk", "walk2"],
        2: [
            "forester",
            "forester2",
            "mustang2",
            "tesla",
            "mustang0528",
            "motor",
            "motor2",
        ],
    },
    "iobt": {1: ["polaris"], 2: ["silverado", "warhog"]},
}
