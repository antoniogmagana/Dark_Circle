import random
import math
import torch
import torchaudio.transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Centralized imports
from db_utils import db_connect, db_close, fetch_sensor_batch, get_time_bounds
import config


class VehicleDataset(Dataset):
    def __init__(self, split):
        self.split = split
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
        dataset, instance, sensor_node, run_id, time, label = self.samples[idx]

        # -----------------------------------------------------------------
        # DYNAMIC SYNTHESIS: Background Augmentation
        # -----------------------------------------------------------------
        # If toggled ON, and the sample is a background class, flip a coin
        if (
            getattr(config, "SYNTHESIZE_BACKGROUND", False)
            and label == 0
            and self.split == "train"
        ):
            if random.random() < getattr(config, "SYNTHESIZE_PROBABILITY", 0.5):
                from data_generator import generate_no_vehicle_sample

                # Use estimated noise floor to create synthetic background samples
                X = generate_no_vehicle_sample(
                    noise_profile="environmental", 
                    amplitude=self.noise_floor
                )
                y = torch.tensor(label, dtype=torch.long)
                return X, y

        sensor_tensors = []

        # Lock to the Global Reference Rate!
        max_time_steps = config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS

        for signal in config.TRAIN_SENSORS:
            exact_table = f"{dataset}_{signal}_{instance}_{sensor_node}"

            # Pass dataset and signal natively down to avoid dangerous string parsing
            sensor_data = self._fetch_sensor_data(
                self.cursor, exact_table, dataset, signal, run_id, time, max_time_steps
            )
            sensor_tensors.append(sensor_data)

        # Concatenate all channels along dim 0
        X = torch.cat(sensor_tensors, dim=0)
        y = torch.tensor(label, dtype=torch.long)

        return X, y

    def _fft_resample(self, signal, target_length):
        """
        Resamples a 1D signal to an exact target length using Fourier interpolation.
        signal: [C, T_in]
        """
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
        sample_rate = config.NATIVE_SR[dataset][signal]
        expected_window = int(sample_rate * config.SAMPLE_SECONDS)

        # Calculate timestamp relative to the synchronized group start time
        min_t = self.table_run_min_time[(table, run_id)]
        start_time_seconds = min_t + float(time * config.SAMPLE_SECONDS)

        # Use the Centralized DB utility!
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

        target_freq = int(max_time_steps / config.SAMPLE_SECONDS)
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
        conn, cursor = db_connect()
        for dataset in config.TRAIN_DATASETS:
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
        temp_conn, temp_cursor = db_connect()

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
                    # Table exists but lacks a run_id column. Rollback and treat as standard table.
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
                signal in present_signals for signal in config.TRAIN_SENSORS
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
                    valid_windows = math.floor(valid_duration / config.SAMPLE_SECONDS)
                    for tr in table_runs:
                        self.table_run_min_time[tr] = group_min_t
                        self.table_run_max_time[tr] = valid_windows

        for k in keys_to_delete:
            del self.table_run_max_time[k]
            del self.table_run_min_time[k]

    def _get_samples(self):
        unique_samples = set()

        block = config.BLOCK_SIZE
        usable = config.USABLE_SIZE

        for (table, run_id), times in self.table_run_max_time.items():
            if times <= 0:
                continue

            parts = table.split("_")
            dataset = parts[0]
            signal = parts[1]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]

            # 1. Fetch from mapping
            category = config.DATASET_VEHICLE_MAP.get(dataset, {}).get(instance, None)
            if category is None:
                continue

            # 2. Assign dynamic label based on config
            if config.TRAINING_MODE == "detection":
                label = 1 if category > 0 else 0
            elif config.TRAINING_MODE == "category":
                label = category
            elif config.TRAINING_MODE == "instance":
                label = config.INSTANCE_TO_CLASS[instance]
            else:
                raise ValueError(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")

            # 3. Block Splitting Logic with Guard Bands
            num_blocks = math.ceil(times / block)
            block_indices = list(range(num_blocks))

            # 4. Handle small-dataset exceptions safely
            try:
                train_blocks, test_blocks = train_test_split(
                    block_indices,
                    test_size=config.SPLIT_TEST + config.SPLIT_VAL,
                    random_state=42,
                )

                val_ratio = config.SPLIT_VAL / (config.SPLIT_TEST + config.SPLIT_VAL)

                # Ensure we have enough test blocks to split into val/test
                if len(test_blocks) > 1:
                    val_blocks, test_blocks = train_test_split(
                        test_blocks, test_size=1.0 - val_ratio, random_state=42
                    )
                else:
                    val_blocks = test_blocks
                    test_blocks = []
                    
            except ValueError:
                # Fallback for recordings too short to split: assign entirely to train
                train_blocks, test_blocks, val_blocks = block_indices, [], []

            # --- START DIAGNOSTIC PRINT ---
            if self.split == "train": # Only print once so it doesn't spam your terminal 3 times
                print(f"DEBUG: [{dataset}] {signal} {instance:<20} (Class {label}) | "
                      f"Total Blocks: {num_blocks:<3} -> "
                      f"Train: {len(train_blocks):<3} | Val: {len(val_blocks):<3} | Test: {len(test_blocks):<3}")
            # --- END DIAGNOSTIC PRINT ---

            # 5. Route the correct blocks to the current dataset instance
            target_blocks = {"train": train_blocks, "test": test_blocks, "val": val_blocks}.get(self.split, [])

            # 5. Route the correct blocks to the current dataset instance
            target_blocks = {"train": train_blocks, "test": test_blocks, "val": val_blocks}.get(self.split, [])

            # 6. Extract the 1-second indices (with guard band enforcement)
            for b_idx in target_blocks:
                start_sec = b_idx * block
                
                # The guard band is enforced here by capping the end_sec
                end_sec = min(start_sec + usable, times) 
                
                for time_idx in range(start_sec, end_sec):
                    unique_samples.add(
                        (dataset, instance, sensor_node, run_id, time_idx, label)
                    )

        self.samples = sorted(list(unique_samples))

        # -----------------------------------------------------------------
        # CONTROLLED OVER-SAMPLING: Background Balancing
        # -----------------------------------------------------------------
        if (
            config.TRAINING_MODE == "detection"
            and self.split == "train"
            and getattr(config, "OVERSAMPLE_BACKGROUNDS", False)
        ):
            background_samples = [s for s in self.samples if s[5] == 0] 
            vehicle_samples = [s for s in self.samples if s[5] == 1]

            shortfall = len(vehicle_samples) - len(background_samples)

            if shortfall > 0:
                import random
                extra_backgrounds = random.choices(background_samples, k=shortfall)
                self.samples.extend(extra_backgrounds)
                random.shuffle(self.samples)

    def close_connection(self):
        if getattr(self, "cursor", None) and not self.cursor.closed:
            self.cursor.close()

        if getattr(self, "conn", None) and self.conn.closed == 0:
            self.conn.close()

    def __del__(self):
        self.close_connection()
