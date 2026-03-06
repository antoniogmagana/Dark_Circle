import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import our custom modular pipeline
from config import DEVICE, BATCH_SIZE, ACOUSTIC_SR, SEISMIC_SR
from db_utils import db_connect, db_close, fetch_sensor_batch, sanitize_name
from data_generator import upsample_seismic_fft, generate_no_vehicle_sample
from preprocess import extract_mel_spectrogram_batch_gpu
from models import ClassificationCNN

# =====================================================================
# 1. SETUP & INITIALIZATION
# =====================================================================
print(f"Initializing Training Pipeline on: {DEVICE}")

# Connect to PostgreSQL
conn, cursor = db_connect()

# 2 channels: Acoustic (1) + Seismic (1) Early Fusion
classifier_model = ClassificationCNN(in_channels=2, num_classes=3).to(DEVICE)
optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dataset map (Vehicle Class -> specific sensor node names)
VEHICLE_MAP = {
    0: {"name": "background", "mic": "mic_01", "geo": "geo_01"},
    1: {"name": "tank", "mic": "mic_01", "geo": "geo_01"},
    2: {"name": "truck", "mic": "mic_01", "geo": "geo_01"}
}

# =====================================================================
# 2. CACHE TABLE SIZES (TIME-BASED)
# =====================================================================
table_max_seconds = {}

def cache_table_durations(cursor, vehicle_map):
    """Finds the maximum safe starting second 'T' for every vehicle."""
    print("Caching table durations for time-aligned sampling...")
    for class_id, v in vehicle_map.items():
        if class_id == 0: 
            continue # Skip caching for background since it's dynamically generated
            
        v_clean = sanitize_name(v["name"])
        mic_clean = sanitize_name(v["mic"])
        table_name = f"audio_16k_{v_clean}_{mic_clean}" 
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            total_rows = cursor.fetchone()[0]
            # Convert total rows to total seconds, subtract 1 for the final window buffer
            total_seconds = (total_rows // ACOUSTIC_SR) - 1 
            table_max_seconds[class_id] = total_seconds
            print(f" - {v['name']}: {total_seconds} seconds available.")
        except Exception as e:
            print(f"Warning: Could not count rows for {table_name}: {e}")
            table_max_seconds[class_id] = 0

cache_table_durations(cursor, VEHICLE_MAP)

# =====================================================================
# 3. THE TIME-ALIGNED BATCH GENERATOR WITH SPLITS
# =====================================================================
def build_random_batch(cursor, vehicle_map, batch_size=BATCH_SIZE, split='train'):
    """
    Pulls a synchronized 1-second window, applies Train/Val/Test splits,
    injects synthetic data for Class 0, and fuses into a Tensor.
    """
    batch_features = []
    batch_labels = []
    available_classes = list(vehicle_map.keys())
    
    while len(batch_features) < batch_size:
        class_id = random.choice(available_classes)
        v_info = vehicle_map[class_id]
        
        # --- SYNTHETIC NO-VEHICLE INJECTION ---
        if class_id == 0:
            np_acoustic = generate_no_vehicle_sample(window_length=ACOUSTIC_SR, noise_profile="environmental", amplitude=0.05)
            np_seismic = generate_no_vehicle_sample(window_length=ACOUSTIC_SR, noise_profile="sensor_hiss", amplitude=0.01)
            
            t_acoustic = torch.from_numpy(np_acoustic)
            t_seismic = torch.from_numpy(np_seismic)
            fused_window = torch.stack((t_acoustic, t_seismic), dim=0)
            
            batch_features.append(fused_window)
            batch_labels.append(class_id)
            continue
            
        # --- REAL VEHICLE DATA FETCH ---
        max_sec = table_max_seconds.get(class_id, 0)
        if max_sec <= 0:
            continue
            
        # Temporal Split Boundaries (70% Train, 15% Val, 15% Test)
        train_bound = int(max_sec * 0.70)
        val_bound = int(max_sec * 0.85)
        
        if split == 'train':
            start_time_sec = random.randint(0, train_bound)
        elif split == 'val':
            start_time_sec = random.randint(train_bound, val_bound)
        elif split == 'test':
            start_time_sec = random.randint(val_bound, max_sec)
            
        acoustic_offset = start_time_sec * ACOUSTIC_SR
        seismic_offset = start_time_sec * SEISMIC_SR
        
        raw_acoustic = fetch_sensor_batch(cursor, "audio_16k", v_info["name"], v_info["mic"], offset=acoustic_offset, limit=ACOUSTIC_SR)
        raw_seismic = fetch_sensor_batch(cursor, "seismic", v_info["name"], v_info["geo"], offset=seismic_offset, limit=SEISMIC_SR)
        
        if not raw_acoustic or not raw_seismic or len(raw_acoustic) < ACOUSTIC_SR or len(raw_seismic) < SEISMIC_SR:
            continue
            
        np_acoustic = np.array([row[0] for row in raw_acoustic], dtype=np.float32)
        np_seismic = np.array([row[0] for row in raw_seismic], dtype=np.float32)
        
        # Upsample seismic to match acoustic dimensions
        np_seismic_upsampled = upsample_seismic_fft(np_seismic, original_sr=SEISMIC_SR, target_sr=ACOUSTIC_SR)
        
        t_acoustic = torch.from_numpy(np_acoustic)
        t_seismic = torch.from_numpy(np_seismic_upsampled)
        fused_window = torch.stack((t_acoustic, t_seismic), dim=0) # Shape: (2, 16000)
        
        batch_features.append(fused_window)
        batch_labels.append(class_id)
        
    batch_tensor = torch.stack(batch_features).to(DEVICE)
    label_tensor = torch.tensor(batch_labels, dtype=torch.long).to(DEVICE)
    
    return batch_tensor, label_tensor

# =====================================================================
# 4. THE TRAINING & VALIDATION LOOP
# =====================================================================
EPOCHS = 10
TRAIN_BATCHES = 50 
VAL_BATCHES = 15

best_val_loss = float('inf')
best_model_weights = copy.deepcopy(classifier_model.state_dict())

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nStarting Training Loop...")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    classifier_model.train()
    running_train_loss, correct_train, total_train = 0.0, 0, 0
    
    for _ in range(TRAIN_BATCHES):
        batch_features, batch_labels = build_random_batch(cursor, VEHICLE_MAP, split='train')
        mel_features = extract_mel_spectrogram_batch_gpu(batch_features, DEVICE)
        
        optimizer.zero_grad()
        outputs = classifier_model(mel_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_labels.size(0)
        correct_train += (predicted == batch_labels).sum().item()

    avg_train_loss = running_train_loss / TRAIN_BATCHES
    train_acc = (correct_train / total_train) * 100
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # --- VALIDATION PHASE ---
    classifier_model.eval()
    running_val_loss, correct_val, total_val = 0.0, 0, 0
    
    with torch.no_grad(): 
        for _ in range(VAL_BATCHES):
            batch_features, batch_labels = build_random_batch(cursor, VEHICLE_MAP, split='val')
            mel_features = extract_mel_spectrogram_batch_gpu(batch_features, DEVICE)
            
            outputs = classifier_model(mel_features)
            loss = criterion(outputs, batch_labels)
            
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += batch_labels.size(0)
            correct_val += (predicted == batch_labels).sum().item()

    avg_val_loss = running_val_loss / VAL_BATCHES
    val_acc = (correct_val / total_val) * 100
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.1f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = copy.deepcopy(classifier_model.state_dict())

classifier_model.load_state_dict(best_model_weights)
print("\nTraining Complete. Best validation weights loaded.")

# Save the optimal weights to disk for future use
torch.save(best_model_weights, "best_multimodal_classifier.pth")

# =====================================================================
# 5. TESTING & VISUALIZATION
# =====================================================================
def evaluate_and_visualize(model, cursor, vehicle_map, test_batches=20):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Starting Final Test Evaluation on Unseen Data...")
    with torch.no_grad():
        for _ in range(test_batches):
            batch_features, batch_labels = build_random_batch(cursor, vehicle_map, split='test')
            mel_features = extract_mel_spectrogram_batch_gpu(batch_features, DEVICE)
            
            outputs = model(mel_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    # Print numerical report
    target_names = [vehicle_map[i]["name"] for i in sorted(vehicle_map.keys())]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Vehicle Classification Confusion Matrix')
    plt.show()

# Run the final evaluation block
evaluate_and_visualize(classifier_model, cursor, VEHICLE_MAP)

# Clean up the PostgreSQL connection
db_close(conn, cursor)