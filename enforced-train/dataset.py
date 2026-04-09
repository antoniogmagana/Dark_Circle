import os
import hashlib
import json
import torch
import torchaudio.transforms
import pandas as pd
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    def __init__(self, split, config):
        self.split = split
        self.config = config
        self.tables = []
        self.table_max_time = {}
        self.split_idx = {}
        self.samples = []
        self.resamplers = {}
        self.reverse_class_map = {v: k for k, v in self.config.CLASS_MAP.items()}

        self._scan_parquet_files()
        self._load_metadata()
        self._align_max_time()
        self._get_samples()
        self._preload_tables()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset, instance, sensor_node, run_id, time, label_str = self.samples[idx]

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

        return X, y, dataset

    def _upsample_signal(self, sensor_data, sample_rate, target_freq):
        resample_key = (sample_rate, target_freq)
        if resample_key not in self.resamplers:
            self.resamplers[resample_key] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_freq
            )
        return self.resamplers[resample_key](sensor_data)

    def _scan_parquet_files(self):
        """Discover all parquet files across DATA_SCAN_DIRS and index by table name."""
        data_dir = getattr(self.config, "DATA_DIR", "../data_files/parsed")
        scan_dirs = getattr(self.config, "DATA_SCAN_DIRS", ["train", "val", "test_iobt"])

        self._parquet_paths = {}  # table_name -> file path

        for subdir in scan_dirs:
            dir_path = os.path.join(data_dir, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if not fname.endswith(".parquet"):
                    continue
                table_name = fname[: -len(".parquet")]
                ds = table_name.split("_")[0]
                if ds not in self.config.TRAIN_DATASETS:
                    continue
                # First occurrence wins — avoids duplicate table keys across subdirs
                if table_name not in self._parquet_paths:
                    self._parquet_paths[table_name] = os.path.join(dir_path, fname)

        self.tables = list(self._parquet_paths.keys())

    def _load_metadata(self):
        """Compute time bounds and present maps in a single read pass per file.

        Replaces the separate _get_table_max_time and _get_present_maps DB calls.
        For m3nvc files the run_id column splits the file into per-run segments,
        matching the multi-run structure that was previously in the database.
        """
        self.table_run_max_time = {}
        self.table_run_min_time = {}
        self.present_maps = {}

        for table in self.tables:
            path = self._parquet_paths[table]
            is_m3nvc = table.startswith("m3nvc_")

            cols = ["time_stamp", "present"]
            if is_m3nvc:
                cols.append("run_id")

            df = pd.read_parquet(path, columns=cols)

            if is_m3nvc:
                for run_id, grp in df.groupby("run_id"):
                    run_id = int(run_id)
                    key = (table, run_id)
                    self.table_run_min_time[key] = float(grp["time_stamp"].min())
                    self.table_run_max_time[key] = float(grp["time_stamp"].max())
                    sec_groups = grp.groupby(grp["time_stamp"].astype(int))["present"].all()
                    self.present_maps[key] = {
                        int(k): bool(v) for k, v in sec_groups.items()
                    }
            else:
                key = (table, None)
                self.table_run_min_time[key] = float(df["time_stamp"].min())
                self.table_run_max_time[key] = float(df["time_stamp"].max())
                sec_groups = df.groupby(df["time_stamp"].astype(int))["present"].all()
                self.present_maps[key] = {
                    int(k): bool(v) for k, v in sec_groups.items()
                }

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

            ref_signal = self.config.TRAIN_SENSORS[0]
            ref_table = f"{dataset}_{ref_signal}_{instance}_{sensor_node}"
            present_map = self.present_maps.get((ref_table, run_id), {})
            is_m3nvc_bg = (dataset == "m3nvc" and instance == "background")

            def _effective_label(step_idx):
                if is_m3nvc_bg:
                    return "background", True
                start_sec = int(step_idx * STEP)
                present_flag = present_map.get(start_sec, False)
                if self.config.TRAINING_MODE == "detection":
                    label = category_str if present_flag else "background"
                    return label, True
                else:
                    return category_str, present_flag

            split_rule = vehicle_info[2]
            WINDOW = self.config.SAMPLE_SECONDS
            STEP = self.config.WINDOW_STEP if self.split == "train" else WINDOW
            valid_duration = times
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
            "data_dir": getattr(self.config, "DATA_DIR", ""),
        }
        key_hash = hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        cache_dir = getattr(self.config, "CACHE_DIR", "cache")
        return os.path.join(cache_dir, f"table_cache_{self.split}_{key_hash}.pt")

    def _preload_tables(self):
        """Bulk-load all required parquet segments into memory tensors.

        Caches to disk so subsequent runs skip the parquet load entirely.
        """
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

        self.table_cache = {}

        needed = set()
        for dataset, instance, sensor_node, run_id, _, _ in self.samples:
            for signal in self.config.TRAIN_SENSORS:
                table = f"{dataset}_{signal}_{instance}_{sensor_node}"
                key = (table, run_id)
                if key in self.table_run_min_time:
                    needed.add(key)

        total = len(needed)
        print(f"[{self.split}] Pre-loading {total} table segments into memory...")

        for i, (table, run_id) in enumerate(sorted(needed)):
            path = self._parquet_paths.get(table)
            if path is None:
                continue

            min_t = self.table_run_min_time[(table, run_id)]

            if "_accel_" in table:
                data_cols = ["accel_x_ew", "accel_y_ns", "accel_z_ud"]
            else:
                data_cols = ["amplitude"]

            read_cols = ["time_stamp"] + data_cols
            if run_id is not None:
                read_cols.append("run_id")

            df = pd.read_parquet(path, columns=read_cols)
            df = df[df["time_stamp"] >= min_t]

            if run_id is not None:
                df = df[df["run_id"] == run_id]

            df = df.sort_values("time_stamp")

            if not df.empty:
                data = torch.tensor(
                    df[data_cols].to_numpy(), dtype=torch.float32
                ).T
                self.table_cache[(table, run_id)] = data

            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(
                    f"  [{self.split}] Loaded {i + 1}/{total} segments...",
                    flush=True,
                )

        print(f"[{self.split}] Pre-loading complete.", flush=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(self.table_cache, cache_path)
        print(f"[{self.split}] Cache saved to {cache_path}", flush=True)
