import os
import time
import csv
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import config
from models import MODEL_REGISTRY
from dataset import VehicleDataset
from train import db_worker_init, compute_stats, compute_noise_floor
from preprocess import preprocess_for_training

def gather_data_into_ram(loader, device, sigma, epsilon, max_samples=None):
    """Rapidly extracts preprocessed 1D waveforms into RAM."""
    X_all, y_all = [], []
    total_samples = 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if max_samples and total_samples >= max_samples:
                break
            x = x.to(device)
            x = preprocess_for_training(x, sigma, epsilon, use_mel=False)
            X_all.append(x.cpu().numpy())
            y_all.append(y.numpy())
            total_samples += x.size(0)
            # if (i + 1) % config.LOG_INTERVAL == 0:
            #     print(f"Data Extraction Progress: {total_samples} samples extracted...")
    
    X_final = np.concatenate(X_all, axis=0)
    y_final = np.concatenate(y_all, axis=0)
    if max_samples:
        return X_final[:max_samples], y_final[:max_samples]
    return X_final, y_final

def main():
    device = config.DEVICE
    print(f"Using device: {device} for preprocessing.")

    # --- ALIGNMENT FIX 1: Create Directory and Save Config Snapshot ---
    print(f"Starting MiniRocket Run ID: {config.RUN_ID}")
    print(f"Saving to: {config.RUN_DIR}")
    config.save_config_snapshot()

    # ------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------
    train_ds = VehicleDataset(split="train")
    val_ds = VehicleDataset(split="val")

    # Estimate sample statistics from 10% sample using law of large numbers
    print(f"Total training samples: {len(train_ds)}")

    calib_size = max(1, int(len(train_ds) * 0.10))
    subset_indices = torch.randperm(len(train_ds))[:calib_size].tolist()
    calib_ds = torch.utils.data.Subset(train_ds, subset_indices)

    # build temp loader to get samples from training set
    calib_loader = DataLoader(calib_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Estimating stats and noise floor from a {calib_size}-sample calibration subset...")

    # Compute global mean/std on the subset
    sigma, epsilon = compute_stats(calib_loader)
    
    # Compute the noise floor (the bottom 5% of amplitudes) on the subset
    noise_floor = compute_noise_floor(calib_loader)

    # store noise floor to training set (val and test sets do not use)
    train_ds.noise_floor = noise_floor

    print(f"Computed Std: {sigma}")
    print(f"Computed Noise Floor (Bottom 5% AC Energy): {noise_floor}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=db_worker_init,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=db_worker_init,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    # ------------------------------------------------------------
    # 2. Calculate Scaling & Save Metadata
    # ------------------------------------------------------------
        
    # Save to the run directory
    torch.save({
        "model_name": config.MODEL_NAME,
        "use_mel": config.USE_MEL,
        "sigma": sigma,
        "epsilon": epsilon
    }, config.META_SAVE_PATH)
    
    print(f"Saved normalization stats to: {config.META_SAVE_PATH}")


    # 3. Extract Data into RAM
    print(f"\n--- Extracting Training Data (Capped at {config.MAX_SAMPLES}) ---")
    X_train, y_train = gather_data_into_ram(train_loader, device, sigma, epsilon, max_samples=config.MAX_SAMPLES)

    print(f"\n--- Extracting Validation Data (Capped at {config.MAX_SAMPLES}) ---")
    X_val, y_val = gather_data_into_ram(val_loader, device, sigma, epsilon, max_samples=config.MAX_SAMPLES)

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