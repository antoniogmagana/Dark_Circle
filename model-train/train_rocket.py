import os
import time
import csv
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import config
from models import MODEL_REGISTRY
from dataset import VehicleDataset
from train import db_worker_init, compute_global_maxs
from preprocess import preprocess_for_training

def gather_data_into_ram(loader, device, channel_maxs, max_samples=None):
    """Rapidly extracts preprocessed 1D waveforms into RAM."""
    X_all, y_all = [], []
    total_samples = 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if max_samples and total_samples >= max_samples:
                break
            x = x.to(device)
            x = preprocess_for_training(x, channel_maxs, use_mel=False)
            X_all.append(x.cpu().numpy())
            y_all.append(y.numpy())
            total_samples += x.size(0)
            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"Data Extraction Progress: {total_samples} samples extracted...")
    
    X_final = np.concatenate(X_all, axis=0)
    y_final = np.concatenate(y_all, axis=0)
    if max_samples:
        return X_final[:max_samples], y_final[:max_samples]
    return X_final, y_final

def main():
    print(f"DEBUG: Run Dir is resolving to -> {os.path.abspath(config.RUN_DIR)}")
    print(f"DEBUG: Model path is resolving to -> {os.path.abspath(config.MODEL_SAVE_PATH)}")

    device = config.DEVICE
    print(f"Using device: {device} for preprocessing.")
    print(f"Starting MiniRocket Run ID: {config.RUN_ID}")

    os.makedirs(config.RUN_DIR, exist_ok=True)

    # 1. Compute or Load Metadata
    if os.path.exists(config.META_SAVE_PATH):
        meta = torch.load(config.META_SAVE_PATH, map_location=device)
        channel_maxs = meta["channel_maxs"]
        print(f"Loaded normalization stats: {channel_maxs.tolist()}")
    else:
        print("Computing global maximums from training set (GPU Accelerated)...")
        temp_ds = VehicleDataset(split="train")
        temp_loader = DataLoader(temp_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        channel_maxs = compute_global_maxs(temp_loader, device)
        torch.save({"channel_maxs": channel_maxs}, config.META_SAVE_PATH)

    # 2. Initialize Training and Validation Datasets ONLY
    train_ds = VehicleDataset(split="train")
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, worker_init_fn=db_worker_init
    )

    val_ds = VehicleDataset(split="val")
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, worker_init_fn=db_worker_init
    )

    # 3. Extract Data into RAM
    print(f"\n--- Extracting Training Data (Capped at {config.ROCKET_MAX_SAMPLES}) ---")
    X_train, y_train = gather_data_into_ram(train_loader, device, channel_maxs, max_samples=config.ROCKET_MAX_SAMPLES)

    print(f"\n--- Extracting Validation Data (Capped at {config.ROCKET_MAX_SAMPLES}) ---")
    X_val, y_val = gather_data_into_ram(val_loader, device, channel_maxs, max_samples=config.ROCKET_MAX_SAMPLES)

    # 4. Train Model
    print(f"\nTraining ClassificationMiniRocket on shape {X_train.shape}...")
    model = MODEL_REGISTRY["ClassificationMiniRocket"]()
    model.fit(X_train, y_train)

    # 5. Save Model
    save_path = config.MODEL_SAVE_PATH.replace(".pth", ".joblib")
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    # 6. Evaluation (Validation Set Only)
    print("\nEvaluating MiniRocket on Validation Set...")
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    # Log to universal metrics tracker
    with open(config.METRICS_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])
        writer.writerow([1, "N/A", f"{train_acc:.4f}", "N/A", f"{val_acc:.4f}"])

    print(f"MiniRocket Fit Summary | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print("Training complete. Run eval_rocket.py for the full Test Set confusion matrix and metrics.")

if __name__ == "__main__":
    main()