import random
import math
import torch
import torchaudio.transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import atexit
from torch.utils.data import get_worker_info

# Centralized imports
from db_utils import db_connect, db_close, fetch_sensor_batch, get_time_bounds


def db_worker_init(worker_id, config=None):
    """
    This function runs once per worker when it is spawned.
    It gives each worker its own dedicated PostgreSQL connection.
    """
    torch.set_num_threads(1)
    worker_info = get_worker_info()
    dataset = worker_info.dataset

    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    # Open the connection and attach it directly to the dataset object
    dataset.conn, dataset.cursor = db_connect(config.DB_CONN_PARAMS)

    atexit.register(dataset.close_connection)


class VehicleDataset(Dataset):
    def __init__(self, split, config):
        self.split = split
        self.config = config
        self.tables = []
        self.table_max_time = {}
        self.split_idx = {}
        self.samples = []
        self.resamplers = {}
        self.conn = None
        self.cursor = None
        self.noise_floor = 0.01

        self._get_tables()
        self._get_table_max_time()
        self._align_max_time()
        self._get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack the fully-defined sample tuple
        dataset, instance, sensor_node, run_id, time, label_str = self.samples[idx]
        
        # --- RESOLVE STRING TO INTEGER ---
        if self.config.TRAINING_MODE == "detection":
            label_int = 0 if label_str == "background" else 1
            
        elif self.config.TRAINING_MODE == "category":
            reverse_class_map = {v: k for k, v in self.config.CLASS_MAP.items()}
            label_int = reverse_class_map.get(label_str, 0) 
            
        elif self.config.TRAINING_MODE == "instance":
            label_int = self.config.INSTANCE_TO_CLASS[instance]
            
        else:
            raise ValueError(f"Unknown TRAINING_MODE: {self.config.TRAINING_MODE}")

        # -----------------------------------------------------------------
        # DYNAMIC SYNTHESIS: Background Augmentation & Pure Synthetic Trap
        # -----------------------------------------------------------------
        is_pure_synthetic = (dataset == "synthetic")
        
        if is_pure_synthetic or (
            getattr(self.config, "SYNTHESIZE_BACKGROUND", False)
            and label_int == 0
            and self.split == "train"
            and random.random() < getattr(self.config, "SYNTHESIZE_PROBABILITY", 0.5)
        ):
            from data_generator import generate_no_vehicle_sample

            # Dynamically select the correct noise floor
            if hasattr(self, "noise_floors") and self.noise_floors:
                # If pure synthetic, fallback to the average of all noise floors
                if is_pure_synthetic or dataset not in self.noise_floors:
                    current_amplitude = torch.stack(list(self.noise_floors.values())).mean(dim=0)
                else:
                    current_amplitude = self.noise_floors[dataset]
            else:
                # Fallback for early initialization before stats are computed
                current_amplitude = getattr(self, "noise_floor", 0.01)

            X = generate_no_vehicle_sample(
                config=self.config,
                noise_profile="environmental", 
                amplitude=current_amplitude
            )
            y = torch.tensor(label_int, dtype=torch.long)
            
            # CRITICAL: Return dataset string here
            return X, y, dataset

        sensor_tensors = []
        max_time_steps = self.config.REF_SAMPLE_RATE * self.config.SAMPLE_SECONDS

        for signal in self.config.TRAIN_SENSORS:
            exact_table = f"{dataset}_{signal}_{instance}_{sensor_node}"

            sensor_data = self._fetch_sensor_data(
                self.cursor, exact_table, dataset, signal, run_id, time, max_time_steps
            )
            sensor_tensors.append(sensor_data)

        X = torch.cat(sensor_tensors, dim=0)
        y = torch.tensor(label_int, dtype=torch.long)

        # CRITICAL: Return dataset string here
        return X, y, dataset

    def _fft_resample(self, signal, target_length):
        C, T_in = signal.shape
        if T_in == target_length:
            return signal

        freqs = torch.fft.rfft(signal, dim=1)
        target_bins = target_length // 2 + 1
        new_freqs = torch.zeros(
            (C, target_bins), dtype=freqs.dtype, device=freqs.device
        )

        copy_bins = min(freqs.shape[1], target_bins)
        new_freqs[:, :copy_bins] = freqs[:, :copy_bins]

        resampled = torch.fft.irfft(new_freqs, n=target_length, dim=1)
        resampled = resampled * (target_length / T_in)

        return resampled

    def _fetch_sensor_data(
        self, cursor, table, dataset, signal, run_id, time, max_time_steps
    ):
        sample_rate = self.config.NATIVE_SR[dataset][signal]
        expected_window = int(sample_rate * self.config.SAMPLE_SECONDS)

        min_t = self.table_run_min_time[(table, run_id)]
        start_time_seconds = min_t + float(time * self.config.SAMPLE_SECONDS)

        raw_data = fetch_sensor_batch(
            cursor=cursor,
            table_name=table,
            sample_count=expected_window,
            start_time=start_time_seconds,
            run_id=run_id,
        )

        if not raw_data:
            run_str = f" (Run: {run_id})" if run_id is not None else ""
            raise ValueError(
                f"CRITICAL: 0 rows returned for {table}{run_str} at time_stamp {start_time_seconds}. "
                f"Check database table for missing rows or alignment issues."
            )

        sensor_data = torch.tensor(raw_data, dtype=torch.float32).T

        if sensor_data.shape[1] != expected_window:
            if hasattr(self, "_fft_resample"):
                sensor_data = self._fft_resample(sensor_data, expected_window)
            else:
                pad_amount = expected_window - sensor_data.shape[1]
                sensor_data = torch.nn.functional.pad(sensor_data, (0, pad_amount))

        target_freq = int(max_time_steps / self.config.SAMPLE_SECONDS)
        if sample_rate < target_freq:
            sensor_data = self._upsample_signal(sensor_data, sample_rate, target_freq)

        if sensor_data.shape[1] > max_time_steps:
            sensor_data = sensor_data[:, :max_time_steps]
        elif sensor_data.shape[1] < max_time_steps:
            pad_amount = max_time_steps - sensor_data.shape[1]
            sensor_data = torch.nn.functional.pad(
                sensor_data, (0, pad_amount), mode="replicate"
            )

        return sensor_data

    def _upsample_signal(self, sensor_data, sample_rate, target_freq):
        resample_key = (sample_rate, target_freq)
        if resample_key not in self.resamplers:
            self.resamplers[resample_key] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_freq
            )
        return self.resamplers[resample_key](sensor_data)

    def _get_tables(self):
        conn, cursor = db_connect(self.config.DB_CONN_PARAMS)
        for dataset in self.config.TRAIN_DATASETS:
            cursor.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname='public'
                  AND tablename LIKE %s;
                  """,
                (f"{dataset}_%",),
            )
            self.tables.extend([table[0] for table in cursor.fetchall()])
        db_close(conn, cursor)

    def _get_table_max_time(self):
        temp_conn, temp_cursor = db_connect(self.config.DB_CONN_PARAMS)

        self.table_run_max_time = {}
        self.table_run_min_time = {}

        for table in self.tables:
            if table.startswith("m3nvc_"):
                try:
                    temp_cursor.execute(
                        f"SELECT DISTINCT run_id FROM {table} WHERE run_id IS NOT NULL;"
                    )
                    runs = [row[0] for row in temp_cursor.fetchall()]
                except Exception:
                    temp_conn.rollback()
                    runs = [None]
            else:
                runs = [None]

            for run_id in runs:
                min_t, max_t = get_time_bounds(temp_cursor, table, run_id=run_id)
                self.table_run_min_time[(table, run_id)] = min_t
                self.table_run_max_time[(table, run_id)] = max_t

        db_close(temp_conn, temp_cursor)

    def _align_max_time(self):
        groups = {}
        for (table, run_id), max_t in self.table_run_max_time.items():
            parts = table.split("_")
            dataset = parts[0]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]

            group_key = (dataset, instance, sensor_node, run_id)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((table, run_id))

        keys_to_delete = []

        for group_key, table_runs in groups.items():
            present_signals = [tr[0].split("_")[1] for tr in table_runs]
            has_all_signals = all(
                signal in present_signals for signal in self.config.TRAIN_SENSORS
            )

            if not has_all_signals:
                for tr in table_runs:
                    keys_to_delete.append(tr)
            else:
                group_min_t = max(self.table_run_min_time[tr] for tr in table_runs)
                group_max_t = min(self.table_run_max_time[tr] for tr in table_runs)

                valid_duration = group_max_t - group_min_t

                if valid_duration <= 0:
                    for tr in table_runs:
                        keys_to_delete.append(tr)
                else:
                    valid_windows = math.floor(valid_duration / self.config.SAMPLE_SECONDS)
                    for tr in table_runs:
                        self.table_run_min_time[tr] = group_min_t
                        self.table_run_max_time[tr] = valid_windows

        for k in keys_to_delete:
            del self.table_run_max_time[k]
            del self.table_run_min_time[k]

    def _get_samples(self):
        unique_samples = set()

        block = self.config.BLOCK_SIZE
        usable = self.config.USABLE_SIZE

        for (table, run_id), times in self.table_run_max_time.items():
            if times <= 0:
                continue

            parts = table.split("_")
            dataset = parts[0]
            signal = parts[1]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]

            # 1. Fetch category STRING from mapping (e.g., "sport")
            category_str = self.config.DATASET_VEHICLE_MAP.get(dataset, {}).get(instance, None)
            
            # If the instance isn't in the map, or if we are in Category mode
            # and the string isn't in our CLASS_MAP (like "background" being skipped), we skip it!
            if category_str is None:
                continue
            if self.config.TRAINING_MODE == "category" and category_str not in self.config.CLASS_MAP.values():
                continue

            # 3. Block Splitting Logic with Guard Bands
            num_blocks = math.ceil(times / block)

            for b_idx in range(num_blocks):
                seed_key = f"{dataset}_{instance}_{sensor_node}_{run_id}_block_{b_idx}"
                rng = random.Random(seed_key)
                rand_val = rng.random()

                if rand_val < self.config.SPLIT_TRAIN:
                    assigned_split = "train"
                elif rand_val < (self.config.SPLIT_TRAIN + self.config.SPLIT_VAL):
                    assigned_split = "val"
                else:
                    assigned_split = "test"

                if assigned_split == self.split:
                    start_sec = b_idx * block
                    end_sec = min(start_sec + usable, times)
                    
                    for time_idx in range(start_sec, end_sec):
                        # WE STORE THE STRING NOW, NOT THE INTEGER
                        unique_samples.add(
                            (dataset, instance, sensor_node, run_id, time_idx, category_str)
                        )

        self.samples = sorted(list(unique_samples))

        # -----------------------------------------------------------------
        # CONTROLLED OVER-SAMPLING: Background Balancing
        # -----------------------------------------------------------------
        if (
            self.config.TRAINING_MODE == "detection"
            and self.split == "train"
            and getattr(self.config, "OVERSAMPLE_BACKGROUNDS", False)
        ):
            # Because we stored strings, we filter by "background"
            background_samples = [s for s in self.samples if s[5] == "background"] 
            vehicle_samples = [s for s in self.samples if s[5] != "background"]

            shortfall = len(vehicle_samples) - len(background_samples)

            if shortfall > 0:
                if len(background_samples) > 0:
                    extra_backgrounds = random.choices(background_samples, k=shortfall)
                    self.samples.extend(extra_backgrounds)
                else:
                    print(f"  [+] Injecting {shortfall} purely synthetic background samples to balance classes.")
                    dummy_samples = [("synthetic", "noise", "none", None, i, "background") for i in range(shortfall)]
                    self.samples.extend(dummy_samples)
                    
                random.shuffle(self.samples)

    def close_connection(self):
        if getattr(self, "cursor", None) and not self.cursor.closed:
            self.cursor.close()

        if getattr(self, "conn", None) and self.conn.closed == 0:
            self.conn.close()

    def __del__(self):
        self.close_connection()