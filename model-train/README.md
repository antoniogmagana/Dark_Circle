## Configuration Guide (`config.py`)

The `config.py` file acts as the central nervous system for the vehicle detection pipeline. It controls database connections, signal processing parameters, model selection, and dynamic label routing. Adjusting these parameters will automatically propagate changes across `train.py`, `eval.py`, `dataset.py`, and `preprocess.py`.

### 1. Model Selection & Architecture

* **`MODEL_NAME`**: The exact class name of the model you want to train (e.g., `"ClassificationCNN"`, `"WaveformClassificationCNN"`, `"ClassificationLSTM"`). This must match a key in `MODEL_REGISTRY` inside `models.py`.
* **`USE_MEL`**: (Auto-computed) Automatically toggles to `True` or `False` based on the selected model's `REQUIRED_SHAPE` class attribute.
* **`LEARNING_RATE`**: (Auto-computed) Automatically pulled from the selected model's `LR` class attribute.
* **`MODEL_SAVE_PATH`**: The output filepath for the best-performing model weights.
* **`META_SAVE_PATH`**: The output filepath for the `channel_maxs` tensor used for global Z-score normalization.

### 2. Training Modes & Objectives

The pipeline supports dynamic label routing. Changing `TRAINING_MODE` automatically re-maps the database `run_id`s to the correct classification task and updates `NUM_CLASSES`.

| Mode | Objective | Description |
| :--- | :--- | :--- |
| `"detection"` | Binary | Classifies samples as either `0` (Background) or `1` (Vehicle). |
| `"category"` | Multi-Class | Classifies by vehicle weight: `0` (Background), `1` (Light), `2` (Heavy). Relies on `CLASS_MAP`. |
| `"instance"` | Specific ID | Treats every unique vehicle name (e.g., `warhog1135am`, `tesla2`) as its own distinct class. |

* **`CLASS_MAP`**: Dictionary mapping numeric IDs to semantic names for category mode.
* **`CLASS_WEIGHTS`**: A list of weights applied to the Cross-Entropy loss function to penalize the model more heavily for missing minority classes (e.g., `[1.0, 10.0, 1.0]`).
* **`INSTANCE_SEED`**: Ensures the randomly shuffled class IDs for "instance" mode remain deterministic across training runs.

### 3. Dataset & Split Parameters

* **`TRAIN_DATASETS`**: List of dataset prefixes to pull from the database (e.g., `["iobt", "focal", "m3nvc"]`).
* **`DATASET_VEHICLE_MAP`**: The master dictionary linking raw database vehicle names to their broader semantic categories.
* **`SPLIT_TRAIN`**: Fraction of time-windows allocated to the training loader (e.g., `0.70`).
* **`SPLIT_VAL`**: Fraction of time-windows allocated to the validation loader (e.g., `0.15`).
* **`SPLIT_TEST`**: Fraction of time-windows allocated to the test loader (e.g., `0.15`).

### 4. Sensor & Time Domain Settings

* **`TRAIN_SENSORS`**: List of sensor modalities to extract (e.g., `["audio", "seismic"]`).
* **`IN_CHANNELS`**: (Auto-computed) The total number of input channels fed to the model, derived from `TRAIN_SENSORS`.
* **`SAMPLE_SECONDS`**: The physical duration of the time window fed into the neural network (e.g., `1` second).
* **`REF_SAMPLE_RATE`**: (Auto-computed) The highest native sample rate across all selected datasets and sensors. Lower-frequency sensors are dynamically upsampled to match this rate in the DataLoader.
* **`NATIVE_SR`**: Dictionary detailing the hardware sample rates of the sensors for each individual dataset.

### 5. Spectrogram & Frequency Settings

These parameters control the math behind `torchaudio.transforms.MelSpectrogram`.

* **`N_FFT`**: Size of the Fast Fourier Transform (e.g., `1024`). Defines the frequency resolution.
* **`HOP_LENGTH`**: Number of audio samples between adjacent STFT columns (e.g., `256`). Controls the time resolution of the 2D output.
* **`MEL_BINS`**: The number of Mel-frequency bands generated (e.g., `64`). Forms the height of the 2D image tensor.
* **`MEL_HOP_LENGTH`** / **`MEL_TOP_DB`**: Auxiliary Mel-scale constraints.

### 6. Training Loop & Hardware Tuning

* **`BATCH_SIZE`**: The number of multi-channel time windows processed simultaneously.
* **`EPOCHS`**: Total passes over the full training dataset.
* **`NUM_WORKERS`**: Number of CPU subprocesses fetching PostgreSQL data. Tuned to maximize database throughput without hitting connection limits.
* **`LOG_INTERVAL`**: How often (in batches) to print loss metrics to the console.
* **`DEVICE`**: (Auto-computed) Automatically binds to CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
* **`DB_CONN_PARAMS`**: Dictionary holding PostgreSQL credentials (`dbname`, `user`, `password`, `host`, `port`).

### 7. Data Augmentation

* **`SYNTHESIZE_BACKGROUND`**: Boolean toggle. When `True`, dynamically generates synthetic environmental noise tensors instead of querying the database for background samples.
* **`SYNTHESIZE_PROBABILITY`**: Float (0.0 to 1.0). The chance that a requested background sample will be synthesized rather than pulled from physical data, acting as infinite data augmentation.

---

## Adding New Datasets (PostgreSQL Schema Guide)

The data pipeline features an **auto-discovery engine** built into `dataset.py`. You do not need to manually hardcode SQL queries for new vehicles or sensor nodes. As long as your tables follow the strict naming conventions and schema below, the pipeline will automatically find, align, and ingest them.

### 1. Table Naming Convention

Every table in the database must follow this exact underscore-separated format:
`[dataset]_[signal]_[instance]_[sensor_node]`

* **`dataset`**: The overarching project or collection name (e.g., `iobt`, `focal`, `m3nvc`).
* **`signal`**: The sensor modality. Must be `audio`, `seismic`, or `accel`.
* **`instance`**: The specific vehicle, event, or background identifier (e.g., `warhog1135am`, `walk`, `mustang`). *Note: Do not use underscores in this name.*
* **`sensor_node`**: The physical hardware unit that recorded the data (e.g., `node1`, `n3`).

**Valid Examples:**

* `iobt_audio_polaris0150pm_node1`
* `focal_seismic_walk_sensor2`
* `m3nvc_accel_cx30_1`

### 2. Required Table Schema (Columns)

The `db_utils.py` fetchers rely on specific column names depending on the `signal` type. All columns storing signal data should be `FLOAT` or `DOUBLE PRECISION`.

**For `audio` and `seismic` tables:**

* `time_stamp` (Float): The absolute or relative time of the reading in seconds. This is critical for multimodal synchronization.
* `amplitude` (Float): The raw signal value.
* `run_id` (Integer, *Optional but Recommended*): Used to separate physically disjointed recordings of the same vehicle into distinct continuous chunks (prevents the dataset from interpolating across massive time gaps).

**For `accel` (3-Axis Accelerometer) tables:**

* `time_stamp` (Float)
* `accel_x_ew` (Float): East-West axis.
* `accel_y_ns` (Float): North-South axis.
* `accel_z_ud` (Float

### 3. Registering the Dataset in `config.py`

Once your tables are injected into PostgreSQL, you only need to update three dictionaries in `config.py` to activate them:

**Step A: Add the prefix to the active training list.**

```python
TRAIN_DATASETS = ["iobt", "focal", "m3nvc", "my_new_dataset"]
```

**Step B: Define the hardware sample rates.**
This tells the `torchaudio` resampler how to normalize the incoming streams.

```python
NATIVE_SR = {
    # ... existing datasets ...
    "my_new_dataset": {"audio": 44100, "seismic": 250, "accel": 250},
}
```

**Step C: Map the instances to semantic categories.**
This links the instance part of your table name to a specific classification category (0 for background, 1 for light, 2 for heavy).

```python
DATASET_VEHICLE_MAP = {
    # ... existing datasets ...
    "my_new_dataset": {
        "quietpark": 0,       # Background
        "hondacivic": 1,      # Light Vehicle
        "dumptruck01": 2      # Heavy Vehicle
    }
}
```

As soon as these three steps are complete, the VehicleDataset will automatically discover the tables via pg_tables, calculate the global temporal intersection across all synchronized modalities, and begin feeding the new tensors directly to the GPU.
