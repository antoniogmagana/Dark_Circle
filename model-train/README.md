## Configuration Guide (`config.py`)

The `config.py` file acts as the central nervous system for the vehicle detection pipeline. It controls database connections, signal processing parameters, model selection, dynamic label routing, and experiment tracking. Adjusting these parameters will automatically propagate changes across `train.py`, `eval.py`, `train_rocket.py`, `eval_rocket.py`, `dataset.py`, and `preprocess.py`.

### 1. Model Selection & Checkpointing
The pipeline dynamically routes inputs and automatically generates unique output directories for every training run.

* **`MODEL_NAME`**: The exact class name of the model you want to train (e.g., `"ClassificationCNN"`, `"WaveformClassificationCNN"`, `"ClassificationMiniRocket"`). This must match a key in `MODEL_REGISTRY` inside `models.py`.
* **`RUN_ID`**: A unique timestamp string (e.g., `20260308_2032`) generated at runtime to prevent files from being overwritten.
* **`RUN_DIR`**: The master directory for a specific experiment, structured as `saved_models/[TRAINING_MODE]/[MODEL_NAME]/[RUN_ID]`.
* **`MODEL_SAVE_PATH`**: The output filepath for the best-performing model weights (`best_model.pth` or `.joblib`).
* **`META_SAVE_PATH`**: The output filepath for the `channel_maxs` tensor used for dynamic Z-score normalization (`meta.pt`).
* **`IMG_SAVE_PATH`**: The output filepath where the evaluation script saves the confusion matrix plot (`conf_matrix.png`).
* **`JSON_LOG_PATH`**: The output filepath for the automated hyperparameter snapshot (`hyperparameters.json`).
* **`METRICS_LOG_PATH`**: The output CSV file tracking epoch-by-epoch loss and accuracy (`metrics.csv`).

### 2. Training Modes & Objectives

The pipeline supports dynamic label routing. Changing `TRAINING_MODE` automatically re-maps the database `run_id`s to the correct classification task and updates `NUM_CLASSES`.

| Mode | Objective | Description |
| :--- | :--- | :--- |
| `"detection"` | Binary | Classifies samples as either `0` (Background) or `1` (Vehicle). |
| `"category"` | Multi-Class | Classifies by vehicle weight: `0` (Background), `1` (Light), `2` (Heavy). Relies on `CLASS_MAP`. |
| `"instance"` | Specific ID | Treats every unique vehicle name (e.g., `warhog1135am`, `tesla2`) as its own distinct class. |

* **`CLASS_MAP`**: Dictionary mapping numeric IDs to semantic names for category mode.
* **`DATASET_VEHICLE_MAP`**: The master dictionary linking raw database vehicle names to their broader semantic categories.
* **`INSTANCE_SEED`**: Ensures the randomly shuffled class IDs for `"instance"` mode remain deterministic across training runs.
* **`CLASS_WEIGHTS`**: A list of weights applied to the Cross-Entropy loss function to penalize the model more heavily for missing minority classes.
* **`ALL_INSTANCES`** / **`INSTANCE_TO_CLASS`**: Lists and dictionaries that dynamically build the label space for instance-level classification.
* **`NUM_CLASSES`**: (Auto-computed) The total number of output nodes required for the final model layer, based on the selected training mode.

### 3. Dataset, Sensor & Split Parameters

* **`TRAIN_DATASETS`**: List of dataset prefixes to pull from the database (e.g., `["iobt", "focal", "m3nvc"]`).
* **`TRAIN_SENSORS`**: List of sensor modalities to extract (e.g., `["audio", "seismic"]`).
* **`IN_CHANNELS`**: (Auto-computed) The total number of input channels fed to the model, derived from `TRAIN_SENSORS`.
* **`SPLIT_TRAIN`** / **`SPLIT_VAL`** / **`SPLIT_TEST`**: Fractions of time-windows allocated to the respective dataloaders (e.g., `0.70`, `0.15`, `0.15`).
* **`NATIVE_SR`**: Dictionary detailing the hardware sample rates of the sensors for each individual dataset.
* **`REF_SAMPLE_RATE`**: (Auto-computed) The highest native sample rate across all selected datasets and sensors.

### 4. Time Domain & Spectrogram Settings

These parameters control the physical time windows and the math behind `torchaudio.transforms.MelSpectrogram`.

* **`SAMPLE_SECONDS`**: The physical duration of the time window fed into the neural network (e.g., `1` second).
* **`N_FFT`**: Size of the Fast Fourier Transform (e.g., `1024`). Defines the frequency resolution.
* **`HOP_LENGTH`**: Number of audio samples between adjacent STFT columns (e.g., `256`).
* **`MEL_BINS`**: The number of Mel-frequency bands generated (e.g., `64`). Forms the height of the 2D image tensor.

### 5. Training Loop & Hardware Tuning

* **`BATCH_SIZE`**: The number of multi-channel time windows processed simultaneously (e.g., `128`).
* **`EPOCHS`**: Total passes over the full training dataset.
* **`NUM_WORKERS`**: Number of CPU subprocesses fetching PostgreSQL data (e.g., `32`).
* **`LOG_INTERVAL`**: How often (in batches) to print loss metrics to the console.
* **`DEVICE`**: (Auto-computed) Automatically binds to CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
* **`DB_CONN_PARAMS`**: Dictionary holding PostgreSQL credentials (`dbname`, `user`, `password`, `host`, `port`).

### 6. Data Augmentation & Dataset Balancing

* **`SYNTHESIZE_BACKGROUND`**: When `True`, dynamically generates synthetic environmental noise tensors instead of using database background samples.
* **`SYNTHESIZE_PROBABILITY`**: Float (0.0 to 1.0). The chance that a requested background sample will be synthesized rather than pulled from physical data.
* **`AUGMENT_SNR`**: When `True`, dynamically injects random noise into the raw waveform during training.
* **`AUGMENT_SNR_RANGE`**: Tuple (Min, Max). The range of Signal-to-Noise Ratio (in decibels) applied during waveform augmentation.
* **`OVERSAMPLE_BACKGROUNDS`**: When `True`, physically duplicates background entries in the dataset index to naturally balance the classes against vehicle samples.

### 7. Model Hyperparameters & Control Flow

Hyperparameters are now explicitly defined per architecture.

* **Base Settings**: `BASE_LR`, `BASE_DROPOUT`.
* **CNN/LSTM Architectures**: Unique channel lists, kernel sizes, strides, paddings, and hidden dimensions for `DET_CNN_*`, `CLASS_CNN_*`, `WAVE_CNN_*`, and `LSTM_*`.
* **MiniRocket**: Uses `ROCKET_NUM_KERNELS`, `ROCKET_ALPHAS`, and `ROCKET_MAX_SAMPLES` to control the scikit-learn Ridge Classifier integration.

### 8. Routing Logic & Experiment Tracking

To prevent circular imports and manual errors, `config.py` uses mapping dictionaries to dictate pipeline behavior based on the chosen `MODEL_NAME`:

* **`LR_MAP`**: Automatically assigns the correct learning rate for the chosen model. Maps to the global `LEARNING_RATE` variable.
* **`SHAPE_MAP`**: Automatically detects if the model requires a `"1D"` (waveform) or `"2D"` (spectrogram) input. This controls the `USE_MEL` boolean passed to the preprocessing engine.
* **`save_config_snapshot()`**: An experiment tracking utility that scans the configuration namespace and dumps all active hyperparameters into a JSON file inside the `RUN_DIR`.

***

## Adding New Datasets (PostgreSQL Schema Guide)

The data pipeline features an **auto-discovery engine** built into `dataset.py`. You do not need to manually hardcode SQL queries for new vehicles or sensor nodes. As long as your tables follow the strict naming conventions and schema below, the pipeline will automatically find, align, and ingest them.

### 1. Table Naming Convention

Every table in the database must follow this exact underscore-separated format:
`[dataset]_[signal]_[instance]_[sensor_node]`

* **dataset**: The overarching project or collection name (e.g., `iobt`, `focal`, `m3nvc`).
* **signal**: The sensor modality. Must be `audio`, `seismic`, or `accel`.
* **instance**: The specific vehicle, event, or background identifier (e.g., `warhog1135am`, `walk`, `mustang`). *Note: Do not use underscores in this name.*
* **sensor_node**: The physical hardware unit that recorded the data (e.g., `node1`, `n3`).

**Valid Examples:**
* `iobt_audio_polaris0150pm_node1`
* `focal_seismic_walk_sensor2`
* `m3nvc_accel_cx30_1`

### 2. Required Table Schema (Columns)

The `db_utils.py` fetchers rely on specific column names depending on the signal type. All columns storing signal data should be `FLOAT` or `DOUBLE PRECISION`.

**For `audio` and `seismic` tables:**
* `time_stamp` (Float): The absolute or relative time of the reading in seconds. This is critical for multimodal synchronization.
* `amplitude` (Float): The raw signal value.
* `run_id` (Integer, *Optional but Recommended*): Used to separate physically disjointed recordings of the same vehicle into distinct continuous chunks.

**For `accel` (3-Axis Accelerometer) tables:**
* `time_stamp` (Float)
* `accel_x_ew` (Float): East-West axis.
* `accel_y_ns` (Float): North-South axis.
* `accel_z_ud` (Float): Up-Down axis.

### 3. Registering the Dataset in `config.py`

Once your tables are injected into PostgreSQL, you only need to update three dictionaries in `config.py` to activate them:

**Step A: Add the prefix to the active training list.**
```python
TRAIN_DATASETS = ["iobt", "focal", "m3nvc", "my_new_dataset"]
```

**Step B: Define the hardware sample rates.**
```python
NATIVE_SR = {
    "my_new_dataset": {"audio": 44100, "seismic": 250, "accel": 250},
}
```
**Step C: Map the instances to semantic categories.**
```python
DATASET_VEHICLE_MAP = {
    "my_new_dataset": {
        "quietpark": 0,       # Background
        "hondacivic": 1,      # Light Vehicle
        "dumptruck01": 2      # Heavy Vehicle
    }
}
```
As soon as these three steps are complete, the VehicleDataset will automatically discover the tables via pg_tables, calculate the global temporal intersection across all synchronized modalities, and begin feeding the new tensors directly to the GPU.
