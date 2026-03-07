import os
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import config
from db_utils import db_connect, get_time_bounds, fetch_sensor_batch
from preprocess import (
    preprocess_for_training,
    preprocess_window,
    extract_mel_spectrogram,
    resample_to,
)
from models import MODEL_REGISTRY


# ============================================================
# 1. Label helpers
# ============================================================


def collate_batch(batch):
    # batch is a list of (window, label)
    xs, ys = zip(*batch)

    # Ensure all windows are contiguous tensors
    xs = [x.contiguous() for x in xs]

    x_batch = torch.stack(xs, dim=0)  # [B, C, T]
    y_batch = torch.tensor(ys, dtype=torch.long)

    return x_batch, y_batch


def extract_instance_from_table(table_name: str) -> str:
    parts = table_name.split("_")
    # dataset = parts[0]
    # signal  = parts[1]
    # instance = parts[2:-1]
    instance_parts = parts[2:-1]
    return "_".join(instance_parts)


def instance_to_category(dataset: str, instance: str) -> int:
    ds_map = config.DATASET_VEHICLE_MAP.get(dataset, {})
    for cat_id, inst_list in ds_map.items():
        if instance in inst_list:
            return cat_id
    raise KeyError(f"Instance '{instance}' not found in dataset '{dataset}' mapping.")


def assign_label(dataset: str, instance: str) -> int:
    category_id = instance_to_category(dataset, instance)

    if config.TRAINING_MODE == "category":
        return category_id

    if config.TRAINING_MODE == "detection":
        return 0 if category_id == 0 else 1

    if config.TRAINING_MODE == "instance":
        return config.INSTANCE_TO_CLASS[instance]

    raise ValueError(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")


# ============================================================
# 2. VehicleStreamer (instance-aware)
# ============================================================


class VehicleStreamer(IterableDataset):
    def __init__(self, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.split = split

        self.chunk_seconds = config.CHUNK_SECONDS
        self.sample_seconds = config.SAMPLE_SECONDS

        self.datasets = config.TRAIN_DATASETS
        self.sensors = config.TRAIN_SENSORS

        # Precompute split thresholds
        self.split_train = config.SPLIT_TRAIN
        self.split_val = config.SPLIT_VAL
        self.split_test = config.SPLIT_TEST

    def _iter_tables(self, cursor, ds_name):
        """
        Yield table names for a given dataset.
        Assumes your DB has tables like: {dataset}_{sensor}_{instance}_rsX
        """
        cursor.execute(
            f"""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE '{ds_name}_%';
            """
        )
        for (tname,) in cursor.fetchall():
            yield tname

    def _choose_split(self, rng_val: float) -> str:
        if rng_val < self.split_train:
            return "train"
        elif rng_val < self.split_train + self.split_val:
            return "val"
        else:
            return "test"

    def __iter__(self):
        conn, cursor = db_connect()
        print("Connected to PostgreSQL successfully.")

        # Infinite stream; DataLoader will stop based on epoch size
        while True:
            # Pick a dataset at random
            ds = random.choice(self.datasets)

            # Get all tables for this dataset
            tables = list(self._iter_tables(cursor, ds))
            if not tables:
                continue

            # Group tables by instance (so we can pick consistent sensor sets)
            instance_to_tables = {}
            for t in tables:
                inst = extract_instance_from_table(t)
                instance_to_tables.setdefault(inst, []).append(t)

            # Pick a random instance
            instance = random.choice(list(instance_to_tables.keys()))
            matching_tables = instance_to_tables[instance]

            # Determine label based on TRAINING_MODE
            label = assign_label(ds, instance)

            # Determine reference sample rate (highest among sensors for this dataset)
            sr_native_ref = max(config.NATIVE_SR[ds][s] for s in self.sensors)

            # For each run, we need time bounds from one representative table
            rep_table = matching_tables[0]
            t_min, t_max = get_time_bounds(cursor, rep_table)
            if t_max <= t_min:
                continue

            total_duration = t_max - t_min
            if total_duration < self.chunk_seconds:
                continue

            # Randomly choose a chunk start time
            win_start = random.uniform(t_min, t_max - self.chunk_seconds)
            win_end = win_start + self.chunk_seconds

            # Decide which split this sample belongs to
            split_choice = self._choose_split(random.random())
            if split_choice != self.split:
                continue

            # --------------------------------------------------
            # Build multi-sensor chunk, resampled to sr_native_ref
            # --------------------------------------------------
            chunk_data = []

            for sensor in self.sensors:
                # Find table for this sensor
                t_name = next((t for t in matching_tables if f"_{sensor}_" in t), None)
                if t_name is None:
                    continue

                sr_native = config.NATIVE_SR[ds][sensor]
                n_samples = int(sr_native * self.chunk_seconds)

                raw = fetch_sensor_batch(
                    cursor,
                    t_name,
                    n_samples,
                    win_start,
                )

                if not raw:
                    continue

                if sensor == "accel":
                    # accel: assume multiple axes already in columns
                    arr = torch.tensor(raw, dtype=torch.float32).T  # [C, T_native]
                else:
                    # audio / seismic: single column
                    arr = torch.tensor([r[0] for r in raw], dtype=torch.float32)[
                        None, :
                    ]  # [1, T_native]

                # Resample to reference rate
                arr = resample_to(arr, sr_native, sr_native_ref)  # [C, T_ref]

                chunk_data.append(arr)

            if not chunk_data:
                continue

            # Concatenate along channel dimension: [C_total, T_ref]
            try:
                chunk = torch.cat(chunk_data, dim=0)
            except RuntimeError:
                # If something is still off, skip this sample
                continue

            # --------------------------------------------------
            # Now slice windows from this chunk
            # --------------------------------------------------
            samples_per_window = int(sr_native_ref * self.sample_seconds)
            total_samples = chunk.shape[-1]

            if total_samples < samples_per_window:
                continue

            # For now, yield a single random window from the chunk
            start_idx = random.randint(0, total_samples - samples_per_window)
            end_idx = start_idx + samples_per_window

            window = chunk[:, start_idx:end_idx]  # [C_total, samples_per_window]

            # Ensure contiguous, normal storage for DataLoader
            window = window.contiguous()

            # Make sure label is a plain int
            label = int(label)

            yield window, label


# ============================================================
# 3. Model loading
# ============================================================


def build_model():
    model_cls = MODEL_REGISTRY[config.MODEL_NAME]

    if config.MODEL_NAME in config.WAVEFORM_ONLY_MODELS:
        use_mel = False
    elif config.MODEL_NAME in config.MEL_ONLY_MODELS:
        use_mel = True
    else:
        use_mel = config.USE_MEL

    model = model_cls(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        use_mel=use_mel,
    )
    return model, use_mel


# ============================================================
# 4. Training + evaluation
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion, device, use_mel):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if config.BATCH_MODE:
            x = preprocess_for_training(x, use_mel=use_mel)
        else:
            x = preprocess_window(x[0]).unsqueeze(0)
            if use_mel:
                x = extract_mel_spectrogram(x)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, use_mel):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if config.BATCH_MODE:
                x = preprocess_for_training(x, use_mel=use_mel)
            else:
                x = preprocess_window(x[0]).unsqueeze(0)
                if use_mel:
                    x = extract_mel_spectrogram(x)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ============================================================
# 5. Main
# ============================================================


def main():
    device = config.DEVICE

    model, use_mel = build_model()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    batch_size = config.BATCH_SIZE if config.BATCH_MODE else 1

    train_loader = DataLoader(
        VehicleStreamer(split="train"),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        VehicleStreamer(split="val"),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_mel
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_mel)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"{config.MODEL_NAME}_{timestamp}_valacc{val_acc:.4f}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "use_mel": use_mel,
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
