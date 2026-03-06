import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from config import (
    DEVICE,
    BATCH_SIZE,
    ACOUSTIC_SR,
    SEISMIC_SR,
    TRAIN_DATASETS,
    TRAIN_SENSORS,
    IN_CHANNELS,
    DATASET_WEIGHTS,
    SENSOR_DROPOUT_PROB,
    CLASS_MAP,
    DATASET_VEHICLE_MAP,
    MODEL_SAVE_PATH,
)
from db_utils import db_connect, db_close, fetch_sensor_batch
from data_generator import upsample_seismic_fft, generate_no_vehicle_sample
from preprocess import extract_mel_spectrogram_batch_gpu, calculate_smv
from models import ClassificationCNN

# =====================================================================
# 1. SETUP & CACHING
# =====================================================================
os.makedirs("../models", exist_ok=True)
print(f"Initializing Training Pipeline on: {DEVICE}")

conn, cursor = db_connect()

model = ClassificationCNN(in_channels=IN_CHANNELS, num_classes=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Cache table lengths to ensure safe random sampling offsets
TABLE_MAX_SECONDS = {}
print("Caching table durations...")
cursor.execute(
    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
)
all_tables = [r[0] for r in cursor.fetchall()]

for t in all_tables:
    if any(t.startswith(ds) for ds in TRAIN_DATASETS):
        cursor.execute(f"SELECT COUNT(*) FROM {t}")
        count = cursor.fetchone()[0]
        sr = ACOUSTIC_SR if "_audio_" in t else SEISMIC_SR
        TABLE_MAX_SECONDS[t] = max(0, (count // sr) - 1)


# =====================================================================
# 2. PARAMETERIZED BATCH GENERATOR
# =====================================================================
def build_parameterized_batch(cursor, batch_size=BATCH_SIZE, split="train"):
    batch_features = []
    batch_labels = []

    ds_choices = [ds for ds in TRAIN_DATASETS if ds in DATASET_WEIGHTS]
    ds_weights = [DATASET_WEIGHTS[ds] for ds in ds_choices]

    while len(batch_features) < batch_size:
        ds = random.choices(ds_choices, weights=ds_weights, k=1)[0]
        class_id = random.choice([0, 1, 2])

        # --- SYNTHETIC BACKGROUND ---
        if class_id == 0:
            syn_channels = []
            if "audio" in TRAIN_SENSORS:
                np_a = generate_no_vehicle_sample(ACOUSTIC_SR, "environmental", 0.05)
                syn_channels.append(torch.from_numpy(np_a))
            if "seismic" in TRAIN_SENSORS or "accel" in TRAIN_SENSORS:
                np_s = generate_no_vehicle_sample(ACOUSTIC_SR, "sensor_hiss", 0.01)
                if "seismic" in TRAIN_SENSORS:
                    syn_channels.append(torch.from_numpy(np_s))
                if "accel" in TRAIN_SENSORS:
                    syn_channels.append(torch.from_numpy(np_s))

            batch_features.append(torch.stack(syn_channels, dim=0))
            batch_labels.append(class_id)
            continue

        # --- REAL VEHICLE DATA ---
        v_base = random.choice(DATASET_VEHICLE_MAP[ds][class_id])
        matching_tables = [t for t in all_tables if f"{ds}_" in t and f"_{v_base}" in t]

        if not matching_tables:
            continue

        # Group by rs node
        nodes = list(set([t.split("_")[-1] for t in matching_tables]))
        rs = random.choice(nodes)

        node_tables = [t for t in matching_tables if t.endswith(rs)]
        audio_t = next((t for t in node_tables if "_audio_" in t), None)
        seis_t = next((t for t in node_tables if "_seismic_" in t), None)
        accel_t = next((t for t in node_tables if "_accel_" in t), None)

        # Verify required tables exist for the requested sensors
        valid = True
        if "audio" in TRAIN_SENSORS and not audio_t:
            valid = False
        if "seismic" in TRAIN_SENSORS and not seis_t:
            valid = False
        if "accel" in TRAIN_SENSORS and not accel_t:
            valid = False
        if not valid:
            continue

        # Temporal Split logic (70/15/15)
        ref_table = audio_t if audio_t else (seis_t if seis_t else accel_t)
        max_sec = TABLE_MAX_SECONDS.get(ref_table, 0)
        if max_sec <= 0:
            continue

        train_bound = int(max_sec * 0.70)
        val_bound = int(max_sec * 0.85)

        if split == "train":
            start_time_sec = random.randint(0, train_bound)
        elif split == "val":
            start_time_sec = random.randint(train_bound, val_bound)
        else:
            start_time_sec = random.randint(val_bound, max_sec)

        current_channels = []

        if "audio" in TRAIN_SENSORS:
            raw_a = fetch_sensor_batch(
                cursor, audio_t, ACOUSTIC_SR, start_time_sec * ACOUSTIC_SR
            )
            current_channels.append(
                torch.from_numpy(np.array([r[0] for r in raw_a], dtype=np.float32))
            )

        if "seismic" in TRAIN_SENSORS:
            raw_s = fetch_sensor_batch(
                cursor, seis_t, SEISMIC_SR, start_time_sec * SEISMIC_SR
            )
            np_s = np.array([r[0] for r in raw_s], dtype=np.float32)
            current_channels.append(
                torch.from_numpy(upsample_seismic_fft(np_s, SEISMIC_SR, ACOUSTIC_SR))
            )

        if "accel" in TRAIN_SENSORS:
            raw_acc = fetch_sensor_batch(
                cursor, accel_t, SEISMIC_SR, start_time_sec * SEISMIC_SR
            )
            if len(raw_acc) == SEISMIC_SR:
                ax, ay, az = zip(*raw_acc)
                smv = calculate_smv(np.array(ax), np.array(ay), np.array(az))
                current_channels.append(
                    torch.from_numpy(upsample_seismic_fft(smv, SEISMIC_SR, ACOUSTIC_SR))
                )
            else:
                # Fallback zero-pad if accel data is corrupted or short
                current_channels.append(torch.zeros(ACOUSTIC_SR, dtype=torch.float32))

        # Sensor Dropout (Only for multi-sensor training in train split)
        if (
            len(TRAIN_SENSORS) > 1
            and split == "train"
            and random.random() < SENSOR_DROPOUT_PROB
        ):
            drop_idx = random.randint(0, len(current_channels) - 1)
            current_channels[drop_idx] = torch.zeros_like(current_channels[drop_idx])

        # Ensure all channels successfully pulled data
        if any(c.shape[0] < ACOUSTIC_SR for c in current_channels):
            continue

        batch_features.append(torch.stack(current_channels))
        batch_labels.append(class_id)

    return torch.stack(batch_features).to(DEVICE), torch.tensor(
        batch_labels, dtype=torch.long
    ).to(DEVICE)


# =====================================================================
# 3. TRAINING LOOP
# =====================================================================
EPOCHS = 10
TRAIN_BATCHES = 50
VAL_BATCHES = 15

best_val_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

print("\nStarting Training Loop...")

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0

    for _ in range(TRAIN_BATCHES):
        batch_features, batch_labels = build_parameterized_batch(cursor, split="train")
        mel_features = extract_mel_spectrogram_batch_gpu(batch_features, DEVICE)

        optimizer.zero_grad()
        outputs = model(mel_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for _ in range(VAL_BATCHES):
            batch_features, batch_labels = build_parameterized_batch(
                cursor, split="val"
            )
            mel_features = extract_mel_spectrogram_batch_gpu(batch_features, DEVICE)
            outputs = model(mel_features)
            loss = criterion(outputs, batch_labels)
            running_val_loss += loss.item()

    avg_t_loss = running_train_loss / TRAIN_BATCHES
    avg_v_loss = running_val_loss / VAL_BATCHES
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f}"
    )

    if avg_v_loss < best_val_loss:
        best_val_loss = avg_v_loss
        best_model_weights = copy.deepcopy(model.state_dict())

# Save optimal weights with metadata
print(f"\nTraining Complete. Saving weights to {MODEL_SAVE_PATH}")
torch.save(
    {
        "model_state_dict": best_model_weights,
        "sensors": TRAIN_SENSORS,
        "datasets": TRAIN_DATASETS,
        "in_channels": IN_CHANNELS,
    },
    MODEL_SAVE_PATH,
)

# =====================================================================
# 4. FINAL EVALUATION (TEST SPLIT)
# =====================================================================
model.load_state_dict(best_model_weights)
model.eval()
all_preds, all_labels = [], []

print("Starting Final Test Evaluation on Unseen Data...")
with torch.no_grad():
    for _ in range(20):
        batch_features, batch_labels = build_parameterized_batch(cursor, split="test")
        mel_features = extract_mel_spectrogram_batch_gpu(batch_features, DEVICE)

        outputs = model(mel_features)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Evaluation restricted to Precision and Recall
precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

target_names = [CLASS_MAP[i] for i in range(3)]
print("\nClassification Evaluation Metrics:")
print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10}")
print("-" * 40)
for i, name in enumerate(target_names):
    print(f"{name:<15} | {precision[i]:<10.4f} | {recall[i]:<10.4f}")

db_close(conn, cursor)
