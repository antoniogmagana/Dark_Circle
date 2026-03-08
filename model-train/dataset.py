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
        dataset, instance, sensor_node, time, label = self.samples[idx]
        sensor_tensors = []

        # Determine the max time steps for this specific group of sensors
        max_time_steps = config.REF_SAMPLE_RATE * config.SAMPLE_SECONDS

        # 1. STRICT CHANNEL ORDERING
        for signal in config.TRAIN_SENSORS:
            # Find the specific table for this sensor type
            exact_table = f"{dataset}_{signal}_{instance}_{sensor_node}"

            # get sensor data
            sensor_data = self._fetch_sensor_data(exact_table, time, max_time_steps)

            sensor_tensors.append(sensor_data)

        # Concatenate all channels along dim 0. Shape will be [IN_CHANNELS, max_time_steps]
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

    def _fetch_sensor_data(self, table, time, max_time_steps):
        parts = table.split("_")
        sample_rate = config.NATIVE_SR[parts[0]][parts[1]]

        expected_window = int(sample_rate * config.SAMPLE_SECONDS)
        sample_offset = int(time * expected_window)

        # Calculate the exact physical timestamp in seconds where this window begins
        start_time_seconds = float(time * config.SAMPLE_SECONDS)

        # Use an indexed WHERE clause to instantly jump to the correct float timestamp
        if parts[1] == "accel":
            query = f"""SELECT accel_x_ew, accel_y_ns, accel_z_ud
                        FROM {table}
                        WHERE time_stamp >= {start_time_seconds}
                        ORDER BY time_stamp ASC
                        LIMIT {expected_window};
                        """
        else:
            query = f"""SELECT amplitude
                        FROM {table}
                        WHERE time_stamp >= {start_time_seconds}
                        ORDER BY time_stamp ASC
                        LIMIT {expected_window};
                        """

        self.cursor.execute(query)
        raw_data = self.cursor.fetchall()

        # Catch completely empty database responses
        if not raw_data:
            raise ValueError(
                f"CRITICAL: 0 rows returned for {table} at time offset {time}. "
                f"Check database table for missing rows or alignment issues."
            )

        # Create Tensor (Shape: [Channels, Length])
        sensor_data = torch.tensor(raw_data, dtype=torch.float32).T

        # 1. FFT Fix: If the database dropped packets, mathematically stretch it
        # to the perfect expected native window using the frequency domain
        if sensor_data.shape[1] != expected_window:
            sensor_data = self._fft_resample(sensor_data, expected_window)

        # 2. Sinc Upsample: Now that the window is perfect, use torchaudio
        # to upsample it to the globally aligned max_time_steps
        target_freq = int(max_time_steps / config.SAMPLE_SECONDS)
        if sample_rate < target_freq:
            sensor_data = self._upsample_signal(sensor_data, sample_rate, target_freq)

        # 3. Final safety clamp (torchaudio can occasionally round off by 1 sample)
        if sensor_data.shape[1] != max_time_steps:
            sensor_data = self._fft_resample(sensor_data, max_time_steps)

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
        _, cursor = db_connect()
        for table in self.tables:
            cursor.execute(f"SELECT count(*) FROM {table};")
            parts = table.split("_")

            self.table_max_time[table] = math.floor(
                cursor.fetchone()[0]
                / config.NATIVE_SR[parts[0]][parts[1]]
                / config.SAMPLE_SECONDS
            )
        db_close(_, cursor)

    def _align_max_time(self):
        groups = {}
        for table, time in self.table_max_time.items():
            parts = table.split("_")
            dataset = parts[0]
            instance = "_".join(parts[2:-1])
            sensor = parts[-1]

            group_key = (dataset, instance, sensor)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(table)

            for group_key, tables in groups.items():
                min_time = min(self.table_max_time[t] for t in tables)
                for t in tables:
                    self.table_max_time[t] = min_time

    def _get_samples(self):
        """
        Determinisitically store indices for train, test,
        validation splits with implied sensor alignment and output labels
        """
        unique_samples = set()

        for table, times in self.table_max_time.items():

            indices = list(range(times))
            parts = table.split("_")
            dataset = parts[0]
            instance = "_".join(parts[2:-1])
            label = config.DATASET_VEHICLE_MAP[parts[0]][instance]
            sensor = parts[-1]

            train, test = train_test_split(indices, test_size=0.3, random_state=42)
            test, val = train_test_split(test, test_size=0.5, random_state=42)

            target_idx = {"train": train, "test": test, "val": val}[self.split]

            for time in target_idx:
                unique_samples.add((dataset, instance, sensor, time, label))

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
