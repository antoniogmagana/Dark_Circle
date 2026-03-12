import os
import torch
from torch.utils.data import DataLoader, get_worker_info
import torch.nn as nn
import torch.optim as optim
import atexit
import csv
from functools import partial

from data_generator import augment_batch
from dataset import VehicleDataset, db_worker_init
from models import build_model
from preprocess import preprocess_for_training
from db_utils import db_connect, db_close

# train.py is the ONLY file that should import the global config directly
import config





def train_one_epoch(
    model, loader, optimizer, criterion, device, sigma, epsilon, config, epoch
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

        # Preprocessing on the GPU (Injecting config)
        x = preprocess_for_training(x, sigma, epsilon, config=config)        

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


def evaluate(model, loader, criterion, device, sigma, epsilon, config):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Preprocessing on the GPU (Injecting config)
            x = preprocess_for_training(x, sigma, epsilon, config=config)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def compute_stats(calib_loader, config):
    """
    Compute the global standard deviation of the AC-coupled training data.
    Global mean is ignored since DC offset is handled at window level.
    """
    channels_sq_sum = torch.zeros(config.IN_CHANNELS)
    total_samples = 0
    
    with torch.no_grad():
        for x, _ in calib_loader:
            # 1. AC-couple the batch
            window_mean = x.mean(dim=-1, keepdim=True)
            x_ac = x - window_mean
            
            # 2. Accumulate the variance of the AC signal
            channels_sq_sum += torch.sum(x_ac**2, dim=[0, 2])
            total_samples += x.shape[0] * x.shape[2]
            
    # Calculate global standard deviation
    sigma = torch.sqrt(channels_sq_sum / total_samples)
    epsilon = 1e-8
    
    return sigma, epsilon


def compute_noise_floor(calib_loader):
    """
    Estimates the natural background amplitude (noise floor) per channel
    by finding the 5th percentile (bottom 5%) of quietest windows in the training subset.
    """
    all_stds = []
    
    with torch.no_grad():
        for x, _ in calib_loader:
            # x shape: [Batch, Channels, Time]
            # Calculate standard deviation (AC amplitude) across the time dimension
            window_stds = torch.std(x, dim=2)  # Shape: [Batch, Channels]
            all_stds.append(window_stds)
            
    # Concatenate all batches: [Total_Subset_Samples, Channels]
    all_stds = torch.cat(all_stds, dim=0)
    
    # Find the 5th percentile of amplitude for each channel.
    # This represents the bottom 5% of the acoustic/seismic energy.
    noise_floor = torch.quantile(all_stds, q=0.05, dim=0)  # Shape: [Channels]
    
    return noise_floor


def main():
    device = config.DEVICE
    print("Using device:", device)

    # --- Create Directory and Save Config Snapshot ---
    print(f"Starting Run ID: {config.RUN_ID}")
    print(f"Saving to: {config.RUN_DIR}")
    config.save_config_snapshot()

    # ------------------------------------------------------------
    # Datasets and loaders (Injecting config)
    # ------------------------------------------------------------
    train_ds = VehicleDataset(split="train", config=config)
    val_ds = VehicleDataset(split="val", config=config)

    # Create partial function for worker init
    custom_worker_init = partial(db_worker_init, config=config)

    print(f"Total training samples: {len(train_ds)}")

    calib_size = max(1, int(len(train_ds) * 0.10))
    subset_indices = torch.randperm(len(train_ds))[:calib_size].tolist()
    calib_ds = torch.utils.data.Subset(train_ds, subset_indices)

    calib_loader = DataLoader(
        calib_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
    )    
    print(f"Estimating stats and noise floor from a {calib_size}-sample calibration subset...")

    # Pass config into compute_stats for IN_CHANNELS
    sigma, epsilon = compute_stats(calib_loader, config=config)
    noise_floor = compute_noise_floor(calib_loader)

    # store noise floor to training set
    train_ds.noise_floor = noise_floor

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    # ------------------------------------------------------------
    # Save Metadata
    # ------------------------------------------------------------
    print(f"Computed Std: {sigma}")
    print(f"Computed Noise Floor: {noise_floor}")
    
    torch.save({
        "model_name": config.MODEL_NAME,
        "use_mel": config.USE_MEL,
        "sigma": sigma,
        "epsilon": epsilon,
        "noise_floor": noise_floor 
    }, config.META_SAVE_PATH)
    
    print(f"Saved normalization stats to: {config.META_SAVE_PATH}")

    # ------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------
    model = build_model(
        input_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES, config=config
    ).to(device)

    print("Performing dummy pass to initialize Lazy modules...")
    model.eval()

    with torch.no_grad():
        for x_dummy, _ in train_loader:
            x_dummy = x_dummy.to(device)
            x_dummy = preprocess_for_training(
                x_dummy, sigma, epsilon, config=config
            )

            if hasattr(model, 'fit_extractor'):
                model.fit_extractor(x_dummy)

            model(x_dummy)
            break

    if len(config.CLASS_WEIGHTS) == config.NUM_CLASSES:
        weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print(
            f"Warning: CLASS_WEIGHTS len ({len(config.CLASS_WEIGHTS)}) != NUM_CLASSES ({config.NUM_CLASSES}). Defaulting to unweighted loss."
        )
        criterion = nn.CrossEntropyLoss()

    optimizer = model.get_optimizer()

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
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
            sigma,
            epsilon,
            config,
            epoch,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            sigma, 
            epsilon,
            config,
        )

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

        with open(config.METRICS_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{train_loss:.4f}", 
                f"{train_acc:.4f}", 
                f"{val_loss:.4f}", 
                f"{val_acc:.4f}"
            ])

    torch.save(
        model.state_dict(),
        config.MODEL_SAVE_PATH,
    )
    print(f"Training complete. Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()