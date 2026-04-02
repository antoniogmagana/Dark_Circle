import os
import hashlib
import json
import random
import torch
import torchaudio.transforms
from torch.utils.data import Dataset
import atexit
from torch.utils.data import get_worker_info

# Centralized imports
from db_utils import (
    db_connect,
    db_close,
    fetch_table_segment,
    get_time_bounds,
    get_present_map,
)


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
        self.noise_floors = None
        self.reverse_class_map = {v: k for k, v in self.config.CLASS_MAP.items()}

        self._get_tables()
        self._get_table_max_time()
        self._align_max_time()
        self._get_present_maps()
        self._get_samples()
        self._preload_tables()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack the fully-defined sample tuple
        dataset, instance, sensor_node, run_id, time, label_str = self.samples[idx]

        # --- RESOLVE STRING TO INTEGER ---
        if self.config.TRAINING_MODE == "detection":
            label_int = 0 if label_str == "background" else 1

        elif self.config.TRAINING_MODE == "category":
            label_int = self.reverse_class_map.get(label_str, 0)

        elif self.config.TRAINING_MODE == "instance":
            vehicle_type = self.config.DATASET_VEHICLE_MAP[dataset][instance][1]
            label_int = self.config.INSTANCE_TO_CLASS[vehicle_type]

        else:
            raise ValueError(f"Unknown TRAINING_MODE: {self.config.TRAINING_MODE}")

        sensor_tensors = []
        max_time_steps = self.config.REF_SAMPLE_RATE * self.config.SAMPLE_SECONDS

        for signal in self.config.TRAIN_SENSORS:
            table = f"{dataset}_{signal}_{instance}_{sensor_node}"
            sample_rate = self.config.NATIVE_SR[dataset][signal]
            expected_window = int(sample_rate * self.config.SAMPLE_SECONDS)

            step = self.config.WINDOW_STEP if self.split == "train" else self.config.SAMPLE_SECONDS
            start = round(time * step * sample_rate)
            cached = self.table_cache[(table, run_id)]
            sensor_data = cached[:, start:start + expected_window].clone()

            if sensor_data.shape[1] < expected_window:
                pad_amount = expected_window - sensor_data.shape[1]
                sensor_data = torch.nn.functional.pad(
                    sensor_data, (0, pad_amount), mode="replicate"
                )

            target_freq = int(max_time_steps / self.config.SAMPLE_SECONDS)
            if sample_rate < target_freq:
                sensor_data = self._upsample_signal(
                    sensor_data, sample_rate, target_freq
                )

            if sensor_data.shape[1] > max_time_steps:
                sensor_data = sensor_data[:, :max_time_steps]

            sensor_tensors.append(sensor_data)

        X = torch.cat(sensor_tensors, dim=0)
        y = torch.tensor(label_int, dtype=torch.long)

        # CRITICAL: Return dataset string here
        return X, y, dataset

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
                if "seismic" in self.config.TRAIN_SENSORS:
                    seismic_trs = [tr for tr in table_runs if tr[0].split("_")[1] == "seismic"]
                    ref_trs = seismic_trs if seismic_trs else table_runs
                else:
                    ref_trs = table_runs
                group_min_t = max(self.table_run_min_time[tr] for tr in ref_trs)
                # End time must be the intersection across all sensors so every
                # sensor's cache covers the full valid_duration window range.
                group_max_t = min(self.table_run_max_time[tr] for tr in table_runs)

                valid_duration = group_max_t - group_min_t

                if valid_duration <= 0:
                    for tr in table_runs:
                        keys_to_delete.append(tr)
                else:
                    for tr in table_runs:
                        self.table_run_min_time[tr] = group_min_t
                        self.table_run_max_time[tr] = valid_duration  # float seconds

        for k in keys_to_delete:
            del self.table_run_max_time[k]
            del self.table_run_min_time[k]

    def _get_present_maps(self):
        """Load per-second present flags for every valid (table, run_id)."""
        conn, cursor = db_connect(self.config.DB_CONN_PARAMS)
        self.present_maps = {}
        for (table, run_id) in self.table_run_max_time:
            self.present_maps[(table, run_id)] = get_present_map(
                cursor, table, run_id=run_id
            )
        db_close(conn, cursor)

    def _get_samples(self):
        unique_samples = set()

        for (table, run_id), times in self.table_run_max_time.items():
            if times < self.config.SAMPLE_SECONDS:
                continue

            parts = table.split("_")
            dataset = parts[0]
            signal = parts[1]
            instance = "_".join(parts[2:-1])
            sensor_node = parts[-1]

            # 1. Fetch category STRING from mapping (e.g., "sport")
            vehicle_info = self.config.DATASET_VEHICLE_MAP.get(dataset, {}).get(
                instance, None
            )
            if vehicle_info is None:
                continue
            category_str = vehicle_info[0]
            if (
                self.config.TRAINING_MODE == "category"
                and category_str not in self.config.CLASS_MAP.values()
            ):
                continue

            # 2. Resolve present map for this (table, run_id)
            # Use the first sensor's table — all sensors share the same
            # time axis so their present maps are identical.
            ref_signal = self.config.TRAIN_SENSORS[0]
            ref_table = f"{dataset}_{ref_signal}_{instance}_{sensor_node}"
            present_map = self.present_maps.get((ref_table, run_id), {})
            is_m3nvc_bg = (dataset == "m3nvc" and instance == "background")

            def _effective_label(step_idx):
                """
                Returns (label_str, include) for a given step index.
                - m3nvc background is always "background" / always included.
                - detection: all windows included; label driven by present_flag.
                - category/instance: only present=True windows included.
                """
                if is_m3nvc_bg:
                    return "background", True
                start_sec = int(step_idx * STEP)
                present_flag = present_map.get(start_sec, False)
                if self.config.TRAINING_MODE == "detection":
                    label = category_str if present_flag else "background"
                    return label, True
                else:
                    return category_str, present_flag

            # 3. Map-Driven Sliding-Window Split Assignment
            split_rule = vehicle_info[2]
            WINDOW = self.config.SAMPLE_SECONDS
            # Sliding windows only for train; eval splits use non-overlapping 1s windows
            STEP = self.config.WINDOW_STEP if self.split == "train" else WINDOW
            valid_duration = times  # float seconds
            max_step_idx = int((valid_duration - WINDOW) / STEP)
            mid_time = valid_duration / 2.0

            def _in_test(step_idx):
                return step_idx * STEP < mid_time

            def _in_val(step_idx):
                return step_idx * STEP >= mid_time

            if split_rule in ("train", "val", "test"):
                if self.split == split_rule:
                    for step_idx in range(max_step_idx + 1):
                        label, include = _effective_label(step_idx)
                        if include:
                            unique_samples.add(
                                (dataset, instance, sensor_node, run_id, step_idx, label)
                            )

            elif split_rule == "split":
                for step_idx in range(max_step_idx + 1):
                    label, include = _effective_label(step_idx)
                    if not include:
                        continue
                    if self.split == "test" and _in_test(step_idx):
                        unique_samples.add(
                            (dataset, instance, sensor_node, run_id, step_idx, label)
                        )
                    elif self.split == "val" and _in_val(step_idx):
                        unique_samples.add(
                            (dataset, instance, sensor_node, run_id, step_idx, label)
                        )

            elif split_rule == "run":
                try:
                    run_id_int = int(run_id)
                except (TypeError, ValueError):
                    continue

                if run_id_int % 2 == 0:
                    if self.split == "train":
                        for step_idx in range(max_step_idx + 1):
                            label, include = _effective_label(step_idx)
                            if include:
                                unique_samples.add(
                                    (dataset, instance, sensor_node, run_id, step_idx, label)
                                )
                else:
                    for step_idx in range(max_step_idx + 1):
                        label, include = _effective_label(step_idx)
                        if not include:
                            continue
                        if self.split == "test" and _in_test(step_idx):
                            unique_samples.add(
                                (dataset, instance, sensor_node, run_id, step_idx, label)
                            )
                        elif self.split == "val" and _in_val(step_idx):
                            unique_samples.add(
                                (dataset, instance, sensor_node, run_id, step_idx, label)
                            )

            else:
                print(
                    f"  [WARN] Unknown split rule '{split_rule}' for "
                    f"{dataset}/{instance}. Skipping."
                )

        self.samples = sorted(list(unique_samples))

    def _get_cache_path(self):
        key_data = {
            "datasets": sorted(self.config.TRAIN_DATASETS),
            "sensors": sorted(self.config.TRAIN_SENSORS),
            "sample_seconds": self.config.SAMPLE_SECONDS,
            "window_step": self.config.WINDOW_STEP,
            "split": self.split,
        }
        key_hash = hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        cache_dir = getattr(self.config, "CACHE_DIR", "cache")
        return os.path.join(cache_dir, f"table_cache_{self.split}_{key_hash}.pt")

    def _preload_tables(self):
        """Bulk-load all required table segments into memory.

        Eliminates per-sample DB queries.
        Caches to disk so subsequent runs skip the DB load entirely."""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            print(
                f"[{self.split}] Loading table cache from disk: {cache_path}",
                flush=True,
            )
            self.table_cache = torch.load(cache_path, weights_only=False)
            print(
                f"[{self.split}] Cache loaded ({len(self.table_cache)} segments).",
                flush=True,
            )
            return

        conn, cursor = db_connect(self.config.DB_CONN_PARAMS)
        self.table_cache = {}

        needed = set()
        for dataset, instance, sensor_node, run_id, _, _ in self.samples:
            if dataset == "synthetic":
                continue
            for signal in self.config.TRAIN_SENSORS:
                table = f"{dataset}_{signal}_{instance}_{sensor_node}"
                key = (table, run_id)
                if key in self.table_run_min_time:
                    needed.add(key)

        total = len(needed)
        print(f"[{self.split}] Pre-loading {total} table segments into memory...")

        for i, (table, run_id) in enumerate(sorted(needed)):
            min_t = self.table_run_min_time[(table, run_id)]
            rows = fetch_table_segment(
                cursor, table, from_time=min_t, run_id=run_id
            )
            if rows:
                self.table_cache[(table, run_id)] = torch.tensor(
                    rows, dtype=torch.float32
                ).T
            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(
                    f"  [{self.split}] Loaded {i + 1}/{total} segments...", flush=True
                )

        db_close(conn, cursor)
        print(f"[{self.split}] Pre-loading complete.", flush=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(self.table_cache, cache_path)
        print(f"[{self.split}] Cache saved to {cache_path}", flush=True)

    def close_connection(self):
        if getattr(self, "cursor", None) and not self.cursor.closed:
            self.cursor.close()

        if getattr(self, "conn", None) and self.conn.closed == 0:
            self.conn.close()

    def __del__(self):
        self.close_connection()
