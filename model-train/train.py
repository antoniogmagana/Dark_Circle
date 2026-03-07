import os
import torch

# =====================================================================
# CRITICAL CPU FIX 1: Prevent "Thread Explosion"
# Forces each worker to use only 1 thread for math, preventing
# 120-core CPUs from spawning thousands of fighting threads.
# =====================================================================
torch.set_num_threads(1)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler, autocast
import random
import copy

import config
import models
import preprocess
from db_utils import db_connect, fetch_sensor_batch, get_time_bounds


class VehicleStreamer(IterableDataset):
    def __init__(self, split="train"):
        super().__init__()
        self.split = split

    def __iter__(self):
        conn, cursor = db_connect()
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
        all_tables = [r[0] for r in cursor.fetchall()]
        bounds_cache = {}

        # =========================================================
        # BULK FETCHING: Fetch 15 seconds of data per DB query
        # =========================================================
        CHUNK_SECONDS = 15

        while True:
            # 1. STOP THE SPIN LOOP: Guaranteed valid class/dataset combos
            class_id = random.choice(list(config.CLASS_MAP.keys()))
            valid_datasets = [
                ds
                for ds in config.TRAIN_DATASETS
                if class_id in config.DATASET_VEHICLE_MAP[ds]
            ]
            if not valid_datasets:
                continue  # Failsafe

            ds = random.choice(valid_datasets)
            v_base = random.choice(config.DATASET_VEHICLE_MAP[ds][class_id])

            try:
                matching_tables = [
                    t for t in all_tables if f"{ds}_" in t and f"_{v_base}" in t
                ]
                if not matching_tables:
                    continue

                nodes = list(set([t.split("_")[-1] for t in matching_tables]))
                rs = random.choice(nodes)

                audio_table = next(
                    (t for t in matching_tables if "_audio_" in t and t.endswith(rs)),
                    None,
                )
                if not audio_table:
                    continue

                target_run_id = (
                    random.choice([8, 9]) if (ds == "m3nvc" and class_id == 0) else None
                )

                cache_key = f"{audio_table}_{target_run_id}"
                if cache_key not in bounds_cache:
                    min_t, max_t = get_time_bounds(
                        cursor, audio_table, run_id=target_run_id
                    )
                    bounds_cache[cache_key] = (min_t, max_t)
                else:
                    min_t, max_t = bounds_cache[cache_key]

                duration = max_t - min_t
                if duration <= CHUNK_SECONDS:
                    continue

                train_bound = min_t + (duration * config.SPLIT_TRAIN)
                val_bound = train_bound + (duration * config.SPLIT_VAL)

                # Adjusted bounds to fit the 15-second chunk safely
                if self.split == "train":
                    high = train_bound - CHUNK_SECONDS
                    if high <= min_t:
                        continue
                    window_start = random.uniform(min_t, high)
                elif self.split == "val":
                    high = val_bound - CHUNK_SECONDS
                    if high <= train_bound:
                        continue
                    window_start = random.uniform(train_bound, high)
                else:
                    high = max_t - CHUNK_SECONDS
                    if high <= val_bound:
                        continue
                    window_start = random.uniform(val_bound, high)

                current_channels = []
                for sensor in config.TRAIN_SENSORS:
                    t_name = next(
                        (
                            t
                            for t in matching_tables
                            if f"_{sensor}_" in t and t.endswith(rs)
                        ),
                        None,
                    )
                    if not t_name:
                        break

                    sensor_type = (
                        "audio"
                        if "audio" in t_name
                        else ("seismic" if "seismic" in t_name else "accel")
                    )
                    sr = config.NATIVE_SR[ds][sensor_type]

                    # 2. BULK QUERY: Fetch exactly CHUNK_SECONDS of data
                    limit_rows = int(sr * CHUNK_SECONDS)
                    raw = fetch_sensor_batch(
                        cursor,
                        t_name,
                        limit_rows,
                        start_time=window_start,
                        run_id=target_run_id,
                    )

                    if not raw or len(raw) < limit_rows:
                        break

                    if "_accel_" in t_name or len(raw[0]) >= 3:
                        x, y, z = [
                            torch.tensor([r[i] for r in raw], dtype=torch.float32)
                            for i in range(3)
                        ]
                        stacked_xyz = torch.stack([x, y, z]).unsqueeze(0)
                        chan_tensor = preprocess.calculate_smv(stacked_xyz).squeeze()
                    else:
                        chan_tensor = torch.tensor(
                            [r[0] for r in raw], dtype=torch.float32
                        )

                    # 3. BULK UPSAMPLE: Stretch to target chunk size
                    target_length = config.ACOUSTIC_SR * CHUNK_SECONDS
                    if chan_tensor.shape[-1] != target_length:
                        chan_tensor = chan_tensor.unsqueeze(0).unsqueeze(0)
                        chan_tensor = preprocess.align_and_upsample(
                            chan_tensor, target_length=target_length
                        ).squeeze()

                    current_channels.append(chan_tensor)

                if len(current_channels) != config.IN_CHANNELS:
                    continue

                # Stack the multi-modal 15-second block
                chunk_tensor = torch.stack(current_channels)

                # 4. YIELD SLICES: Loop over the chunk locally in RAM
                for i in range(CHUNK_SECONDS):
                    start_idx = i * config.ACOUSTIC_SR
                    end_idx = start_idx + config.ACOUSTIC_SR
                    yield chunk_tensor[:, start_idx:end_idx], class_id

            except Exception as e:
                continue


def run_training(model_class):
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    model = model_class().to(config.DEVICE)

    if isinstance(model, models.ClassificationCNN):
        lr = models.CLASS_CNN_LR
    elif isinstance(model, models.DetectionCNN):
        lr = models.DET_CNN_LR
    else:
        lr = models.BASE_LR

    # =====================================================================
    # CRITICAL CPU FIX 3: Unleash the Hardware
    # Now that the CPU isn't thrashing, we can spin up 24 safe workers
    # to blast your 1 TB of RAM with prefetched data.
    # =====================================================================
    train_loader = DataLoader(
        VehicleStreamer("train"),
        batch_size=config.BATCH_SIZE,
        num_workers=24,
        pin_memory=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        VehicleStreamer("val"),
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(models.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(config.DEVICE, non_blocking=True), labels.to(
                config.DEVICE, non_blocking=True
            )
            features = preprocess.align_and_upsample(
                preprocess.zero_center_window(features)
            )

            if isinstance(model, (models.ClassificationCNN, models.DetectionCNN)):
                features = preprocess.extract_mel_spectrogram(features)

            with autocast(
                device_type=config.DEVICE.type, enabled=config.DEVICE.type != "cpu"
            ):
                outputs = model(features)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            if i >= config.TRAIN_STEPS_PER_EPOCH - 1:
                break

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (v_features, v_labels) in enumerate(val_loader):
                v_features = preprocess.align_and_upsample(
                    preprocess.zero_center_window(
                        v_features.to(config.DEVICE, non_blocking=True)
                    )
                )
                if isinstance(model, (models.ClassificationCNN, models.DetectionCNN)):
                    v_features = preprocess.extract_mel_spectrogram(v_features)

                v_outputs = model(v_features)
                total_val_loss += criterion(
                    v_outputs, v_labels.to(config.DEVICE, non_blocking=True)
                ).item()
                if i >= config.VAL_STEPS_PER_EPOCH - 1:
                    break

        avg_v_loss = total_val_loss / config.VAL_STEPS_PER_EPOCH
        print(
            f"Epoch {epoch} | Train Loss: {total_train_loss/config.TRAIN_STEPS_PER_EPOCH:.4f} | Val Loss: {avg_v_loss:.4f}"
        )

        if avg_v_loss < best_val_loss:
            best_val_loss, best_weights = avg_v_loss, copy.deepcopy(model.state_dict())
            print(f"--> Best model saved at epoch {epoch}")

    if best_weights:
        torch.save(
            {
                "model_state_dict": best_weights,
                "sensors": config.TRAIN_SENSORS,
                "datasets": config.TRAIN_DATASETS,
                "in_channels": config.IN_CHANNELS,
                "val_loss": best_val_loss,
            },
            config.MODEL_SAVE_PATH,
        )


if __name__ == "__main__":
    run_training(models.ClassificationCNN)
