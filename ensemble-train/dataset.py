import random
import math
import torch
import torchaudio.transforms
from torch.utils.data import Dataset, DataLoader, get_worker_info
from functools import partial
import atexit

from db_utils import db_connect, db_close, fetch_sensor_batch, get_time_bounds


def db_worker_init(worker_id, config=None):
    """Give each DataLoader worker its own PostgreSQL connection."""
    torch.set_num_threads(1)
    worker_info = get_worker_info()
    dataset = worker_info.dataset

    # Unwrap Subset wrapper if present
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    dataset.conn, dataset.cursor = db_connect(config.DB_CONN_PARAMS)
    atexit.register(dataset.close_connection)


# =====================================================================
# In-Memory Dataset (serves cached data — no DB access)
# =====================================================================

class MemoryDataset(Dataset):
    """
    Wraps pre-loaded tensors in memory.  Created by preload_to_memory().

    Data can live on CPU or GPU.  When on GPU, the training loop's
    .to(device) call is a no-op — zero transfer cost per batch.

    Synthetic background injection still runs at access time so
    augmentation remains dynamic across epochs.
    """

    def __init__(self, X, y, datasets, config, split):
        self.X = X
        self.y = y
        self.datasets = datasets
        self.config = config
        self.split = split
        self.noise_floors = {}
        self.device = X.device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_val = self.y[idx]
        dataset = self.datasets[idx]

        label_int = y_val.item()

        if dataset == "synthetic" or self._should_synthesize(label_int):
            return self._make_synthetic_sample(dataset, label_int)

        return x, y_val, dataset

    def _should_synthesize(self, label_int):
        return (
            getattr(self.config, "SYNTHESIZE_BACKGROUND", False)
            and label_int == 0
            and self.split == "train"
            and random.random() < getattr(self.config, "SYNTHESIZE_PROBABILITY", 0.5)
        )

    def _make_synthetic_sample(self, dataset, label_int):
        from data_generator import generate_no_vehicle_sample

        if self.noise_floors:
            if dataset not in self.noise_floors or dataset == "synthetic":
                amplitude = torch.stack(list(self.noise_floors.values())).mean(dim=0)
            else:
                amplitude = self.noise_floors[dataset]
        else:
            amplitude = 0.01

        X = generate_no_vehicle_sample(
            config=self.config,
            noise_profile="environmental",
            amplitude=amplitude,
        )
        # Move synthetic sample to same device as the rest of the data
        if self.device.type != "cpu":
            X = X.to(self.device)
        y = torch.tensor(label_int, dtype=torch.long, device=self.device)
        return X, y, dataset


# Maximum dataset size (MB) to place on GPU.
# Seismic ~300MB, accel ~1GB fit easily.  Audio ~25GB won't.
_GPU_CACHE_LIMIT_MB = 4096


def preload_to_memory(db_dataset, config, device=None):
    """
    Load all samples from a VehicleDataset into RAM via one full pass.

    If the data fits within _GPU_CACHE_LIMIT_MB and a CUDA device is
    available, tensors are moved directly to GPU — eliminating all
    CPU→GPU transfer during training.
    """
    custom_init = partial(db_worker_init, config=config)
    loader = DataLoader(
        db_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_init,
    )

    all_X, all_y, all_ds = [], [], []

    for batch_idx, (x, y, dataset_names) in enumerate(loader):
        all_X.append(x)
        all_y.append(y)
        all_ds.extend(dataset_names)

        done = sum(t.shape[0] for t in all_X)
        if (batch_idx + 1) % 100 == 0:
            print(f"    {done:,} / {len(db_dataset):,} samples loaded...", flush=True)

    X = torch.cat(all_X, dim=0)
    y = torch.cat(all_y, dim=0)

    size_mb = (X.nelement() * X.element_size()) / (1024 ** 2)

    # Move to GPU if it fits
    if device is None:
        device = config.DEVICE if hasattr(config, "DEVICE") else torch.device("cpu")

    if device.type == "cuda" and size_mb < _GPU_CACHE_LIMIT_MB:
        X = X.to(device)
        y = y.to(device)
        print(f"    Cached {len(y):,} samples on GPU ({size_mb:,.0f} MB)")
    else:
        # Pin memory for faster async CPU→GPU transfer
        X = X.pin_memory()
        y = y.pin_memory()
        location = "pinned RAM" if device.type == "cuda" else "RAM"
        print(f"    Cached {len(y):,} samples in {location} ({size_mb:,.0f} MB)")

    mem_ds = MemoryDataset(X, y, all_ds, config, db_dataset.split)
    mem_ds.noise_floors = db_dataset.noise_floors
    return mem_ds


# =====================================================================
# Database-Backed Dataset
# =====================================================================

# Dictionary keys throughout this class use the tuple:
#   (table, run_id, scene_id)
#
# For iobt and focal: run_id=None, scene_id=None
# For m3nvc: run_id=int, scene_id=int
#
# Sample tuples stored in self.samples:
#   (dataset, instance, sensor_node, run_id, scene_id, time_idx, category_str)


class VehicleDataset(Dataset):
    def __init__(self, split, config):
        self.split = split
        self.config = config
        self.tables = []
        self.table_max_time = {}   # (table, run_id, scene_id) → valid_windows
        self.table_min_time = {}   # (table, run_id, scene_id) → aligned_min_t
        self.samples = []
        self.resamplers = {}
        self.conn = None
        self.cursor = None

        # Will be populated externally by train.py after calibration
        self.noise_floors = {}

        self._discover_tables()
        self._compute_time_bounds()
        self._align_sensor_groups()
        self._build_samples()

    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------------------
    # Sample retrieval
    # -----------------------------------------------------------------

    def __getitem__(self, idx):
        dataset, instance, sensor_node, run_id, scene_id, time_idx, label_str = self.samples[idx]

        label_int = self._resolve_label(dataset, instance, label_str)

        # Synthetic background path
        if dataset == "synthetic" or self._should_synthesize(label_int):
            return self._make_synthetic_sample(dataset, label_int)

        # Real data path
        max_time_steps = self.config.REF_SAMPLE_RATE * self.config.SAMPLE_SECONDS
        sensor_tensors = []

        for signal in self.config.TRAIN_SENSORS:
            exact_table = f"{dataset}_{signal}_{instance}_{sensor_node}"
            sensor_data = self._fetch_sensor_data(
                self.cursor, exact_table, dataset, signal,
                run_id, scene_id, time_idx, max_time_steps
            )
            sensor_tensors.append(sensor_data)

        X = torch.cat(sensor_tensors, dim=0)
        y = torch.tensor(label_int, dtype=torch.long)
        return X, y, dataset

    def _resolve_label(self, dataset, instance, label_str):
        """Convert string label to integer based on training mode."""
        mode = self.config.TRAINING_MODE
        if mode == "detection":
            return 0 if label_str == "background" else 1
        elif mode == "category":
            reverse_map = {v: k for k, v in self.config.CLASS_MAP.items()}
            return reverse_map.get(label_str, 0)
        elif mode == "instance":
            return self.config.INSTANCE_TO_CLASS[instance]
        raise ValueError(f"Unknown TRAINING_MODE: {mode}")

    def _should_synthesize(self, label_int):
        """Check whether this background sample should be replaced with synthetic data."""
        return (
            getattr(self.config, "SYNTHESIZE_BACKGROUND", False)
            and label_int == 0
            and self.split == "train"
            and random.random() < getattr(self.config, "SYNTHESIZE_PROBABILITY", 0.5)
        )

    def _make_synthetic_sample(self, dataset, label_int):
        """Generate a synthetic background sample with calibrated noise floor."""
        from data_generator import generate_no_vehicle_sample

        if self.noise_floors:
            if dataset not in self.noise_floors or dataset == "synthetic":
                amplitude = torch.stack(list(self.noise_floors.values())).mean(dim=0)
            else:
                amplitude = self.noise_floors[dataset]
        else:
            amplitude = 0.01

        X = generate_no_vehicle_sample(
            config=self.config,
            noise_profile="environmental",
            amplitude=amplitude,
        )
        y = torch.tensor(label_int, dtype=torch.long)
        return X, y, dataset

    # -----------------------------------------------------------------
    # Sensor data fetching & resampling
    # -----------------------------------------------------------------

    def _fetch_sensor_data(self, cursor, table, dataset, signal,
                           run_id, scene_id, time_idx, max_time_steps):
        sample_rate = self.config.NATIVE_SR[dataset][signal]
        expected_window = int(sample_rate * self.config.SAMPLE_SECONDS)

        min_t = self.table_min_time[(table, run_id, scene_id)]
        start_time = min_t + float(time_idx * self.config.SAMPLE_SECONDS)

        raw_data = fetch_sensor_batch(
            cursor=cursor,
            table_name=table,
            sample_count=expected_window,
            start_time=start_time,
            run_id=run_id,
            scene_id=scene_id,
        )

        if not raw_data:
            run_str = f" (run={run_id}, scene={scene_id})" if run_id is not None else ""
            raise ValueError(
                f"CRITICAL: 0 rows for {table}{run_str} at time_stamp {start_time}. "
                f"Check database for missing rows or alignment issues."
            )

        sensor_data = torch.tensor(raw_data, dtype=torch.float32).T

        # Pad if DB returned fewer rows than expected
        if sensor_data.shape[1] != expected_window:
            sensor_data = self._fft_resample(sensor_data, expected_window)

        # Upsample to the reference sample rate if needed
        target_freq = int(max_time_steps / self.config.SAMPLE_SECONDS)
        if sample_rate < target_freq:
            sensor_data = self._upsample_signal(sensor_data, sample_rate, target_freq)

        # Final length enforcement
        if sensor_data.shape[1] > max_time_steps:
            sensor_data = sensor_data[:, :max_time_steps]
        elif sensor_data.shape[1] < max_time_steps:
            pad = max_time_steps - sensor_data.shape[1]
            sensor_data = torch.nn.functional.pad(sensor_data, (0, pad), mode="replicate")

        return sensor_data

    def _fft_resample(self, signal, target_length):
        """Resample via FFT zero-padding / truncation."""
        C, T_in = signal.shape
        if T_in == target_length:
            return signal

        freqs = torch.fft.rfft(signal, dim=1)
        target_bins = target_length // 2 + 1
        new_freqs = torch.zeros(C, target_bins, dtype=freqs.dtype)

        copy_bins = min(freqs.shape[1], target_bins)
        new_freqs[:, :copy_bins] = freqs[:, :copy_bins]

        resampled = torch.fft.irfft(new_freqs, n=target_length, dim=1)
        return resampled * (target_length / T_in)

    def _upsample_signal(self, sensor_data, sample_rate, target_freq):
        key = (sample_rate, target_freq)
        if key not in self.resamplers:
            self.resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_freq
            )
        return self.resamplers[key](sensor_data)

    # -----------------------------------------------------------------
    # Table discovery & time bounds
    # -----------------------------------------------------------------

    def _discover_tables(self):
        """Find all tables in the DB matching the configured datasets."""
        conn, cursor = db_connect(self.config.DB_CONN_PARAMS)
        for dataset in self.config.TRAIN_DATASETS:
            cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' AND tablename LIKE %s;",
                (f"{dataset}_%",),
            )
            self.tables.extend(row[0] for row in cursor.fetchall())
        db_close(conn, cursor)

    def _compute_time_bounds(self):
        """
        Get min/max timestamps per (table, run_id, scene_id) triple.

        For m3nvc: each (scene_id, run_id) pair is a separate recording.
        For iobt/focal: run_id=None, scene_id=None (one recording per table).
        """
        conn, cursor = db_connect(self.config.DB_CONN_PARAMS)

        for table in self.tables:
            if table.startswith("m3nvc_"):
                try:
                    cursor.execute(
                        f"SELECT DISTINCT scene_id, run_id FROM {table} "
                        f"WHERE scene_id IS NOT NULL AND run_id IS NOT NULL;"
                    )
                    recordings = [(row[0], row[1]) for row in cursor.fetchall()]
                except Exception:
                    conn.rollback()
                    recordings = [(None, None)]
            else:
                recordings = [(None, None)]

            for scene_id, run_id in recordings:
                min_t, max_t = get_time_bounds(
                    cursor, table, run_id=run_id, scene_id=scene_id
                )
                key = (table, run_id, scene_id)
                self.table_min_time[key] = min_t
                self.table_max_time[key] = max_t

        db_close(conn, cursor)

    # -----------------------------------------------------------------
    # Cross-sensor alignment
    # -----------------------------------------------------------------

    def _align_sensor_groups(self):
        """
        Group tables by (dataset, instance, sensor_node, run_id, scene_id)
        and align their time windows.

        Alignment is computed across ALL co-located sensors listed in
        ALIGN_SENSORS (default: ["audio", "seismic"]).  This guarantees
        that training on audio and training on seismic produce identical
        sample lists, even though each run only uses one sensor's data.

        A group is valid if:
          1. All ALIGN_SENSORS are present (for consistent window counts)
          2. The TRAIN_SENSOR is present (so we can actually fetch data)
        """
        align_sensors = getattr(self.config, "ALIGN_SENSORS", ["audio", "seismic"])

        groups = {}
        for (table, run_id, scene_id), max_t in self.table_max_time.items():
            parts = table.split("_")
            dataset = parts[0]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]
            key = (dataset, instance, sensor_node, run_id, scene_id)
            groups.setdefault(key, []).append((table, run_id, scene_id))

        keys_to_delete = []

        for group_key, table_keys in groups.items():
            present_signals = [tk[0].split("_")[1] for tk in table_keys]

            has_align = all(s in present_signals for s in align_sensors)
            has_train = all(s in present_signals for s in self.config.TRAIN_SENSORS)

            if not (has_align and has_train):
                keys_to_delete.extend(table_keys)
                continue

            # Time intersection across ALL sensors in the group
            group_min = max(self.table_min_time[tk] for tk in table_keys)
            group_max = min(self.table_max_time[tk] for tk in table_keys)
            duration = group_max - group_min

            if duration <= 0:
                keys_to_delete.extend(table_keys)
                continue

            valid_windows = math.floor(duration / self.config.SAMPLE_SECONDS)
            for tk in table_keys:
                self.table_min_time[tk] = group_min
                self.table_max_time[tk] = valid_windows

        for k in keys_to_delete:
            self.table_max_time.pop(k, None)
            self.table_min_time.pop(k, None)

    # -----------------------------------------------------------------
    # Sample list construction
    # -----------------------------------------------------------------

    def _build_samples(self):
        """
        Build the list of sample tuples:
          (dataset, instance, sensor_node, run_id, scene_id, time_idx, category_str)
        """
        unique_samples = set()
        block = self.config.BLOCK_SIZE
        usable = self.config.USABLE_SIZE
        train_sensor = self.config.TRAIN_SENSORS[0]

        for (table, run_id, scene_id), total_windows in self.table_max_time.items():
            if total_windows <= 0:
                continue

            # Only build samples from the sensor we're training on
            parts = table.split("_")
            table_signal = parts[1]
            if table_signal != train_sensor:
                continue

            dataset = parts[0]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]

            category_str = self.config.DATASET_VEHICLE_MAP.get(dataset, {}).get(instance)
            if category_str is None:
                continue
            if (
                self.config.TRAINING_MODE == "category"
                and category_str not in self.config.CLASS_MAP.values()
            ):
                continue

            num_blocks = math.ceil(total_windows / block)
            for b_idx in range(num_blocks):
                # scene_id in the seed ensures different scenes get independent splits
                seed_key = (
                    f"{dataset}_{instance}_{sensor_node}"
                    f"_{run_id}_{scene_id}_block_{b_idx}"
                )
                rng = random.Random(seed_key)
                rand_val = rng.random()

                if rand_val < self.config.SPLIT_TRAIN:
                    assigned_split = "train"
                elif rand_val < self.config.SPLIT_TRAIN + self.config.SPLIT_VAL:
                    assigned_split = "val"
                else:
                    assigned_split = "test"

                if assigned_split != self.split:
                    continue

                start_sec = b_idx * block
                end_sec = min(start_sec + usable, total_windows)
                for t in range(start_sec, end_sec):
                    unique_samples.add(
                        (dataset, instance, sensor_node,
                         run_id, scene_id, t, category_str)
                    )

        self.samples = sorted(unique_samples)
        self._balance_backgrounds()

    def _balance_backgrounds(self):
        """Over-sample or inject synthetic backgrounds for detection mode."""
        if not (
            self.config.TRAINING_MODE == "detection"
            and self.split == "train"
            and getattr(self.config, "OVERSAMPLE_BACKGROUNDS", False)
        ):
            return

        bg = [s for s in self.samples if s[6] == "background"]
        fg = [s for s in self.samples if s[6] != "background"]
        shortfall = len(fg) - len(bg)

        if shortfall <= 0:
            return

        if bg:
            self.samples.extend(random.choices(bg, k=shortfall))
        else:
            print(f"  [+] Injecting {shortfall} synthetic background samples.")
            synthetic = [
                ("synthetic", "noise", "none", None, None, i, "background")
                for i in range(shortfall)
            ]
            self.samples.extend(synthetic)

        random.shuffle(self.samples)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def close_connection(self):
        if getattr(self, "cursor", None) and not self.cursor.closed:
            self.cursor.close()
        if getattr(self, "conn", None) and self.conn.closed == 0:
            self.conn.close()

    def __del__(self):
        self.close_connection()