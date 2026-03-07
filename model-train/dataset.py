import random
import torch
from torch.utils.data import Dataset
from db_utils import db_connect, fetch_sensor_batch, get_time_bounds
import config


class VehicleDataset(Dataset):
    """
    Finite, leak‑free dataset with per‑instance splitting.
    Each item is a 1‑second multimodal window:
        [audio, seismic, accel_x, accel_y, accel_z]
    """

    def __init__(self, split):
        assert split in {"train", "val", "test"}
        self.split = split

        conn, cursor = db_connect()

        # ============================================================
        # 1. Discover all tables and group them by instance
        # ============================================================
        instance_to_tables = {}  # inst → [table1, table2, ...]
        instance_to_dataset = {}  # inst → dataset name

        for ds in config.TRAIN_DATASETS:
            cursor.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname='public'
                  AND tablename LIKE %s;
            """,
                (f"{ds}_%",),
            )
            tables = [t[0] for t in cursor.fetchall()]

            for t in tables:
                # Example: iobt_audio_polaris0150pm_rs3
                parts = t.split("_")
                # parts = [dataset, sensor, instance, rsX]
                if len(parts) < 4:
                    continue

                dataset = parts[0]
                signal = parts[1]
                inst = "_".join(parts[2:-1])
                sensor = parts[-1]

                instance_to_tables.setdefault(inst, []).append(t)
                instance_to_dataset[inst] = dataset

        # ============================================================
        # 2. Compute valid 1‑second windows per instance
        # ============================================================
        instance_to_windows = {}

        for inst, tables in instance_to_tables.items():
            ds = instance_to_dataset[inst]

            # Pick any table for time bounds
            rep_table = tables[0]
            t_min, t_max = get_time_bounds(cursor, rep_table)
            duration_sec = int(t_max - t_min)

            if duration_sec <= 0:
                continue

            # Build window list
            instance_to_windows[inst] = [(ds, inst, sec) for sec in range(duration_sec)]

        # ============================================================
        # 3. Per‑instance split (no leakage)
        # ============================================================
        rng = random.Random(config.INSTANCE_SEED)
        all_instances = list(instance_to_windows.keys())
        rng.shuffle(all_instances)

        n = len(all_instances)
        n_train = int(0.7 * n)
        n_val = int(0.2 * n)

        if split == "train":
            chosen_instances = all_instances[:n_train]
        elif split == "val":
            chosen_instances = all_instances[n_train : n_train + n_val]
        else:
            chosen_instances = all_instances[n_train + n_val :]

        # Flatten windows for chosen instances
        self.index = []
        for inst in chosen_instances:
            self.index.extend(instance_to_windows[inst])

        # Store for __getitem__
        self.instance_to_tables = instance_to_tables
        self.instance_to_dataset = instance_to_dataset
        self.ref_sr = config.REF_SAMPLE_RATE

        self.conn = conn
        self.cursor = cursor

    # ============================================================
    # PyTorch Dataset API
    # ============================================================

    def set_normalization(self, sensor_mins, sensor_maxs):
        self.sensor_mins = sensor_mins
        self.sensor_maxs = sensor_maxs

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ds, inst, sec = self.index[idx]
        tables = self.instance_to_tables[inst]

        chunk_data = []

        for sensor in config.TRAIN_SENSORS:
            # Find the correct table for this sensor (rsX suffix varies)
            matches = [t for t in tables if f"_{sensor}_" in t]
            if not matches:
                # Missing sensor → zero channel
                arr = torch.zeros(1, int(self.ref_sr), dtype=torch.float32)
                chunk_data.append(arr)
                continue

            table = matches[0]
            sr_native = config.NATIVE_SR[ds][sensor]

            # Fetch exactly 1 second
            raw = fetch_sensor_batch(self.cursor, table, sr_native, sec)

            if not raw:
                arr = torch.zeros(1, int(self.ref_sr), dtype=torch.float32)
            else:
                if sensor == "accel":
                    # raw is list of rows: [ [x,y,z], ... ]
                    arr = torch.tensor(raw, dtype=torch.float32).T  # [3, T_native]
                else:
                    # raw is list of single-column rows
                    arr = torch.tensor([r[0] for r in raw], dtype=torch.float32)[
                        None, :
                    ]

                # Resample to global reference rate
                arr = resample_to(arr, sr_native, self.ref_sr)

                # -----------------------------
                # Apply min/max normalization
                # -----------------------------
                min_v = self.sensor_mins[sensor]
                max_v = self.sensor_maxs[sensor]

                # Avoid divide-by-zero
                arr = (arr - min_v) / (max_v - min_v + 1e-8)  # [0,1]
                arr = arr * 2 - 1  # [-1,1]

                chunk_data.append(arr)

        # Concatenate channels: [audio, seismic, accel_x, accel_y, accel_z]
        window = torch.cat(chunk_data, dim=0)

        # Label is determined by dataset + instance
        label = assign_label(ds, inst)

        return window, label


# ============================================================
# Helper functions you already have elsewhere
# ============================================================


def resample_to(arr, sr_native, sr_ref):
    """
    Resample arr from sr_native → sr_ref.
    arr: [C, T_native]
    """
    import torchaudio

    return torchaudio.functional.resample(arr, sr_native, sr_ref)


def assign_label(dataset_name, instance_name):
    """
    Your existing logic for mapping (dataset, instance) → class label.
    """
    return config.INSTANCE_LABELS[(dataset_name, instance_name)]
