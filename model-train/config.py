import os
import torch

# --- HARDWARE CONFIGURATION ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
else:
    DEVICE = torch.device("cpu")

# --- DATABASE CONFIGURATION ---
DB_CONN_PARAMS = {
    "dbname": "lvc_db",
    "user": "lvc_toolkit",
    "password": os.environ.get("DB_PASSWORD", "default_password_if_any"),
    "host": "localhost",
    "port": 5432,
}

# Fetch size: Exactly 64 windows at 16kHz
DB_CHUNK_SIZE = 1024000  

# --- MACHINE LEARNING HYPERPARAMETERS ---
BATCH_SIZE = 64         

# Sample Rates (Hz)
ACOUSTIC_SR = 16000  
ACOUSTIC_SR2 = 1600  
SEISMIC_SR = 100     
SEISMIC_SR2 = 200