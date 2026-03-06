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
TRAIN_DATASETS = ["iobt", "focal", "m3nvc"]
TRAIN_SENSORS = ["audio", "seismic"]
IN_CHANNELS = len(TRAIN_SENSORS)

# The TARGET sample rate all tensors will be upsampled to for the CNN
ACOUSTIC_SR = 16000

# --- NEW: Native Sample Rates per Dataset ---
NATIVE_SR = {
    "iobt": {"audio": 16000, "seismic": 100, "accel": 100},
    "focal": {"audio": 16000, "seismic": 100, "accel": 100},
    "m3nvc": {"audio": 1600, "seismic": 200, "accel": 200},
}

CLASS_MAP = {0: "background", 1: "light", 2: "heavy"}

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
# 4. REPEATED SETTINGS & ORCHESTRATION CONTROLS
# =====================================================================
MODEL_SAVE_PATH = "saved_models/best_vehicle_model.pth"

BATCH_SIZE = 128
TRAIN_STEPS_PER_EPOCH = 50
VAL_STEPS_PER_EPOCH = 16
EVAL_STEPS = 50

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

MEL_BINS = 64
MEL_HOP_LENGTH = 512
MEL_TOP_DB = 80
NOISE_KERNEL_SIZE = 51
