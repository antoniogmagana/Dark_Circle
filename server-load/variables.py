import os

# where the files are stored
MOD_path = "~/datasets/MOD_vehicle"
M3NVC_path = "~/datasets/M3NVC"

# Database connection settings
conn_params = {
    "dbname": "lvc_db",
    "user": "lvc_toolkit",
    "password": os.environ.get("DB_PASSWORD", "default_password_if_any"),
    "host": "localhost",
    "port": 5432,
}

# sample rates
ACOUSTIC_PR = 0.0000625  # period for 16 kHz
ACOUSTIC_PR2 = 0.000625  # period for 1.6 kHz
SEISMIC_PR = 0.01  # period for 100 Hz
SEISMIC_PR2 = 0.005  # period for 200 Hz

# processing chunk size
CHUNK_SIZE = 500000  # As requested
