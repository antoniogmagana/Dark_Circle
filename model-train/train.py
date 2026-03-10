import os
import torch
from torch.utils.data import DataLoader, get_worker_info
import torch.nn as nn
import torch.optim as optim
import atexit
import csv

from data_generator import augment_batch
from dataset import VehicleDataset
from models import build_model
from preprocess import preprocess_for_training
from db_utils import db_connect, db_close
import config


def db_worker_init(worker_id):
    """
    This function runs once per worker when it is spawned.
    It gives each worker its own dedicated PostgreSQL connection.
    """
    torch.set_num_threads(1)
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # Get this worker's copy of the dataset

    # Open the connection and attach it directly to the dataset object
    dataset.conn, dataset.cursor = db_connect()

    atexit.register(dataset.close_connection)


def train_one_epoch(
    model, loader, optimizer, criterion, device, mu, sigma, epsilon, use_mel, epoch
):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # -----------------------------------------------------------------
        # DYNAMIC SYNTHESIS: SNR Augmentation (Controlled via config)
        # -----------------------------------------------------------------
        if getattr(config, "AUGMENT_SNR", False):
            # Safely grab the range, defaulting to (10, 30) if missing
            current_snr_range = getattr(config, "AUGMENT_SNR_RANGE", (10, 30))
            x = augment_batch(x, snr_range=current_snr_range)
        # -----------------------------------------------------------------

        # Preprocessing on the GPU
        x = preprocess_for_training(x, mu, sigma, epsilon, use_mel)        

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

        # Logging
        # if (batch_idx + 1) % config.LOG_INTERVAL == 0:
        #     print(
        #         f"Train Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
        #         f"| Loss: {loss.item():.4f}",
        #         flush=True,
        #     )

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, mu, sigma, epsilon, use_mel):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Preprocessing on the GPU (Same scaling as training!)
            x = preprocess_for_training(x, mu, sigma, epsilon, use_mel=use_mel)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def compute_stats(train_loader):
    """
    Compute mean and standard deviation of training data.
    Assumes 3-channel data. 
    """
    channels_sum = torch.zeros(config.IN_CHANNELS) 
    channels_sq_sum = torch.zeros(config.IN_CHANNELS)
    num_batches = len(train_loader)
    
    for x, y in train_loader:
        # leaving stats per channel
        channels_sum += torch.mean(x, dim=[0, 2])
        channels_sq_sum += torch.mean(x**2, dim=[0, 2])
    
    # Divide by the total number of batches, not the size of the last batch
    mean = channels_sum / num_batches
    std = (channels_sq_sum / num_batches - mean**2).sqrt()
    epsilon = 1e-8
    
    return mean, std, epsilon



def main():
    device = config.DEVICE
    print("Using device:", device)

    # --- NEW: Create Directory and Save Config Snapshot ---
    print(f"Starting Run ID: {config.RUN_ID}")
    print(f"Saving to: {config.RUN_DIR}")
    config.save_config_snapshot()

    # ------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------
    train_ds = VehicleDataset(split="train")
    val_ds = VehicleDataset(split="val")

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
    # Calculate Scaling & Save Metadata
    # ------------------------------------------------------------
    print("Computing normalization statistics from training data...")
    mu, sigma, epsilon = compute_stats(train_loader)
    print(f"Computed Mean: {mu}")
    print(f"Computed Std: {sigma}")
    
    # Save to the run directory
    torch.save({
        "model_name": config.MODEL_NAME,
        "use_mel": config.USE_MEL,
        "mu": mu,
        "sigma": sigma,
        "epsilon": epsilon
    }, config.META_SAVE_PATH)
    
    print(f"Saved normalization stats to: {config.META_SAVE_PATH}")

    # ------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------
    model = build_model(
        input_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES
    ).to(device)

    # 1. Robust Fix: Dummy pass to initialize PyTorch Lazy modules before optimizer binding
    print("Performing dummy pass to initialize Lazy modules...")
    model.eval()
    with torch.no_grad():
        for x_dummy, _ in train_loader:
            x_dummy = x_dummy.to(device)
            x_dummy = preprocess_for_training(
                x_dummy, mu, sigma, epsilon, use_mel=config.USE_MEL
            )
            model(x_dummy)
            break

    # 2. Robust Fix: Dynamic weight allocation check
    if len(config.CLASS_WEIGHTS) == config.NUM_CLASSES:
        weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print(
            f"Warning: CLASS_WEIGHTS len ({len(config.CLASS_WEIGHTS)}) != NUM_CLASSES ({config.NUM_CLASSES}). Defaulting to unweighted loss."
        )
        criterion = nn.CrossEntropyLoss()

    # Optimizer must be defined AFTER the dummy pass
    optimizer = model.get_optimizer()

# ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    # 1. Initialize the CSV file with headers
    with open(config.METRICS_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            mu,
            sigma,
            epsilon,
            config.USE_MEL,
            epoch,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            mu, sigma, epsilon,
            config.USE_MEL,
        )

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

        # 2. Append the current epoch's metrics to the CSV
        with open(config.METRICS_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{train_loss:.4f}", 
                f"{train_acc:.4f}", 
                f"{val_loss:.4f}", 
                f"{val_acc:.4f}"
            ])

    # ------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------
    torch.save(
        model.state_dict(),
        config.MODEL_SAVE_PATH,
    )
    print(f"Training complete. Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
