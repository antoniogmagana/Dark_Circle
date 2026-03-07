import os
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import config
from db_utils import db_connect, get_time_bounds, fetch_sensor_batch
from preprocess import (
    preprocess_for_training,
    preprocess_window,
    extract_mel_spectrogram,
)
from models import MODEL_REGISTRY


# ============================================================
# 1. Label helpers
# ============================================================


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
    def __init__(self, split="train"):
        super().__init__()
        self.split = split
        self.chunk_seconds = config.CHUNK_SECONDS
        self.sample_seconds = config.SAMPLE_SECONDS

    def __iter__(self):
        conn, cursor = db_connect()

        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public'"
        )
        all_tables = [r[0] for r in cursor.fetchall()]

        bounds_cache = {}

        while True:
            # -----------------------------
            # Pick dataset
            # -----------------------------
            ds = random.choice(config.TRAIN_DATASETS)
            ds_tables = [t for t in all_tables if t.startswith(ds + "_")]
            if not ds_tables:
                continue

            # -----------------------------
            # Pick a table and extract instance
            # -----------------------------
            table_name = random.choice(ds_tables)
            instance = extract_instance_from_table(table_name)

            # -----------------------------
            # Extract signal + sensor id
            # -----------------------------
            parts = table_name.split("_")
            signal = parts[1]
            sensor_id = parts[-1]

            # -----------------------------
            # Find all tables for this instance + sensor
            # -----------------------------
            matching_tables = [
                t
                for t in ds_tables
                if extract_instance_from_table(t) == instance and t.endswith(sensor_id)
            ]
            if not matching_tables:
                continue

            # -----------------------------
            # Cache time bounds
            # -----------------------------
            for t_name in matching_tables:
                if t_name not in bounds_cache:
                    bounds_cache[t_name] = get_time_bounds(cursor, t_name)

            try:
                min_t = max(bounds_cache[t][0] for t in matching_tables)
                max_t = min(bounds_cache[t][1] for t in matching_tables)
            except Exception:
                continue

            if max_t - min_t <= self.chunk_seconds:
                continue

            # -----------------------------
            # Random chunk start
            # -----------------------------
            win_start = random.uniform(min_t, max_t - self.chunk_seconds)

            # -----------------------------
            # Fetch chunk for each sensor
            # -----------------------------
            chunk_data = []
            sr_native_ref = None

            for sensor in config.TRAIN_SENSORS:
                t_name = next(
                    (t for t in matching_tables if f"_{sensor}_" in t),
                    None,
                )
                if t_name is None:
                    continue

                sr_native = config.NATIVE_SR[ds][sensor]
                if sr_native_ref is None:
                    sr_native_ref = sr_native

                raw = fetch_sensor_batch(
                    cursor,
                    t_name,
                    int(sr_native * self.chunk_seconds),
                    win_start,
                )

                if sensor == "accel":
                    arr = torch.tensor(raw, dtype=torch.float32).T
                else:
                    arr = torch.tensor([r[0] for r in raw], dtype=torch.float32)[
                        None, :
                    ]

                chunk_data.append(arr)

            if not chunk_data:
                continue

            chunk = torch.cat(chunk_data, dim=0)

            # -----------------------------
            # Slice into 1-second window
            # -----------------------------
            samples_per_window = int(sr_native_ref * self.sample_seconds)
            num_windows = int(self.chunk_seconds // self.sample_seconds)
            if samples_per_window <= 0 or num_windows <= 0:
                continue

            w = random.randint(0, num_windows - 1)
            start = w * samples_per_window
            end = start + samples_per_window
            if end > chunk.shape[-1]:
                continue

            window = chunk[:, start:end]

            # -----------------------------
            # Random split assignment
            # -----------------------------
            p = random.random()
            if p < config.SPLIT_TRAIN:
                assigned_split = "train"
            elif p < config.SPLIT_TRAIN + config.SPLIT_VAL:
                assigned_split = "val"
            else:
                assigned_split = "test"

            if assigned_split != self.split:
                continue

            # -----------------------------
            # Assign label
            # -----------------------------
            label = assign_label(ds, instance)

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
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        VehicleStreamer(split="val"),
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
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
