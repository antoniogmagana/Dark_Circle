import random
import math
import torch
import torchaudio.transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from db_utils import db_connect, db_close, fetch_sensor_batch, get_time_bounds
import config


class VehicleDataset(Dataset):
    """ """

    def __init__(self, split):
        self.split = split
        self.tables = []
        self.table_max_time = {}
        self.split_idx = {}
        self.samples = []
        self.resamplers = {}
        self.conn = None
        self.cursor = None

        self._get_tables()
        self._get_table_max_time()
        self._align_max_time()
        self._get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack the fully-defined sample tuple
        dataset, instance, sensor_node, run_id, time, label = self.samples[idx]
        sensor_tensors = []

        # Lock to the Global Reference Rate!
        max_time_steps = config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS

        for signal in config.TRAIN_SENSORS:
            exact_table = f"{dataset}_{signal}_{instance}_{sensor_node}"

            # Pass run_id down to the database query
            sensor_data = self._fetch_sensor_data(
                self.cursor, exact_table, run_id, time, max_time_steps
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

        # 1. Transform to frequency domain
        freqs = torch.fft.rfft(signal, dim=1)

        # 2. Create empty frequency bins for the new target length
        target_bins = target_length // 2 + 1
        new_freqs = torch.zeros(
            (C, target_bins), dtype=freqs.dtype, device=freqs.device
        )

        # 3. Map frequencies (truncates if downsampling, zero-pads if upsampling)
        copy_bins = min(freqs.shape[1], target_bins)
        new_freqs[:, :copy_bins] = freqs[:, :copy_bins]

        # 4. Inverse transform back to the time domain
        resampled = torch.fft.irfft(new_freqs, n=target_length, dim=1)

        # 5. Scale amplitude to preserve original energy levels
        # Scaling factor: N_target / N_input
        resampled = resampled * (target_length / T_in)

        return resampled

    def _fetch_sensor_data(self, cursor, table, run_id, time, max_time_steps):
        parts = table.split("_")
        sample_rate = config.NATIVE_SR[parts[0]][parts[1]]

        expected_window = int(sample_rate * config.SAMPLE_SECONDS)

        # We now calculate the timestamp relative to the perfectly synchronized group start time
        min_t = self.table_run_min_time[(table, run_id)]
        start_time_seconds = min_t + float(time * config.SAMPLE_SECONDS)

        if run_id is not None:
            where_clause = (
                f"WHERE run_id = {run_id} AND time_stamp >= {start_time_seconds}"
            )
        else:
            where_clause = f"WHERE time_stamp >= {start_time_seconds}"

        if parts[1] == "accel":
            query = f"""SELECT accel_x_ew, accel_y_ns, accel_z_ud
                        FROM {table}
                        {where_clause}
                        ORDER BY time_stamp ASC
                        LIMIT {expected_window};
                        """
        else:
            query = f"""SELECT amplitude
                        FROM {table}
                        {where_clause}
                        ORDER BY time_stamp ASC
                        LIMIT {expected_window};
                        """

        cursor.execute(query)
        raw_data = cursor.fetchall()

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
        # pull resample structure from cache
        resample_key = (sample_rate, target_freq)

        # If resampler not built yet, build it and save it
        if resample_key not in self.resamplers:
            self.resamplers[resample_key] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_freq
            )

        sensor_data = self.resamplers[resample_key](sensor_data)

        return sensor_data

    def _get_tables(self):
        _, cursor = db_connect()
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
        db_close(_, cursor)

    def _get_table_max_time(self):
        # 1. Open a temporary connection
        temp_conn, temp_cursor = db_connect()

        self.table_run_max_time = {}
        self.table_run_min_time = (
            {}
        )  # NEW: We must track where the recording actually started

        for table in self.tables:
            if table.startswith("m3nvc_"):
                temp_cursor.execute(
                    f"SELECT DISTINCT run_id FROM {table} WHERE run_id IS NOT NULL;"
                )
                runs = [row[0] for row in temp_cursor.fetchall()]
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
            # 1. COMPLETENESS CHECK: Ensure every requested sensor exists for this run
            present_signals = [tr[0].split("_")[1] for tr in table_runs]
            has_all_signals = all(
                signal in present_signals for signal in config.TRAIN_SENSORS
            )

            if not has_all_signals:
                # If a modality is completely missing, discard this run so it is never sampled
                for tr in table_runs:
                    keys_to_delete.append(tr)
            else:
                # 2. TIME INTERSECTION: Find the synchronized absolute time overlap
                group_min_t = max(self.table_run_min_time[tr] for tr in table_runs)
                group_max_t = min(self.table_run_max_time[tr] for tr in table_runs)

                valid_duration = group_max_t - group_min_t

                if valid_duration <= 0:
                    for tr in table_runs:
                        keys_to_delete.append(tr)
                else:
                    # Apply the synchronized bounds back to the tables
                    valid_windows = math.floor(valid_duration / config.SAMPLE_SECONDS)
                    for tr in table_runs:
                        self.table_run_min_time[tr] = group_min_t
                        self.table_run_max_time[tr] = valid_windows

        for k in keys_to_delete:
            del self.table_run_max_time[k]
            del self.table_run_min_time[k]

    def _get_samples(self):
        unique_samples = set()

        for (table, run_id), times in self.table_run_max_time.items():
            indices = list(range(times))
            parts = table.split("_")
            dataset = parts[0]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]
            label = config.DATASET_VEHICLE_MAP[dataset][instance]

            train, test = train_test_split(indices, test_size=0.3, random_state=42)
            test, val = train_test_split(test, test_size=0.5, random_state=42)

            target_idx = {"train": train, "test": test, "val": val}[self.split]

            for time in target_idx:
                # Store the FULL identity of the sample, including run_id!
                unique_samples.add(
                    (dataset, instance, sensor_node, run_id, time, label)
                )

        self.samples = sorted(list(unique_samples))

    def close_connection(self):
        """Safely closes DB connections if they exist and are open."""
        # Check if cursor exists and isn't already closed
        if getattr(self, "cursor", None) and not self.cursor.closed:
            self.cursor.close()

        # Check if conn exists and isn't already closed (psycopg2 uses closed == 0 for open)
        if getattr(self, "conn", None) and self.conn.closed == 0:
            self.conn.close()

    def __del__(self):
        """Python's garbage collection destructor."""
        self.close_connection()
