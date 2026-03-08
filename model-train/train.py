import torch
from torch.utils.data import DataLoader, get_worker_info
import torch.nn as nn
import torch.optim as optim
import atexit

from dataset import VehicleDataset
from models import build_model
from preprocess import preprocess_for_training, extract_mel_spectrogram
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
    model, loader, optimizer, criterion, device, channel_maxs, use_mel, epoch
):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # Preprocessing on the GPU
        x = preprocess_for_training(x, channel_maxs, use_mel=use_mel)

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
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(
                f"Train Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True,
            )

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, channel_maxs, use_mel):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Preprocessing on the GPU (Same scaling as training!)
            x = preprocess_for_training(x, channel_maxs, use_mel=use_mel)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def compute_global_maxs(train_loader, device):
    """
    Runs a single pass over the training data to find the absolute maximum
    amplitude per channel, AFTER zero-centering the windows.
    Returns a tensor of shape [C] containing the max values.
    """
    print("Computing global maximums from training set (GPU Accelerated)...")
    channel_maxs = None
    total_batches = len(train_loader)

    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)

            # 1. Zero-center the windows first (subtract mean along the time dimension)
            x_centered = x - x.mean(dim=-1, keepdim=True)

            # 2. Find the max absolute value for each channel in this batch
            # amax(dim=(0, 2)) checks across the Batch and Time dimensions, leaving just Channels
            batch_maxs = x_centered.abs().amax(dim=(0, 2))

            # 3. Keep the running maximums
            if channel_maxs is None:
                channel_maxs = batch_maxs
            else:
                channel_maxs = torch.maximum(channel_maxs, batch_maxs)

            # NEW: Print progress on batches
            if (i + 1) % config.LOG_INTERVAL == 0 or i == total_batches - 1:
                print(
                    f"Max Computation Progress: Batch {i+1}/{total_batches}", flush=True
                )

    print(f"Global Channel Maxs found: {channel_maxs.cpu().tolist()}")
    return channel_maxs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
        worker_init_fn=db_worker_init,  # FIXED TYPO: was worker_int_fn
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=db_worker_init,  # FIXED TYPO: was worker_int_fn
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    # ------------------------------------------------------------
    # Calculate Global Maxes for Scaling
    # ------------------------------------------------------------
    # We do this AFTER train_loader is built, but BEFORE the training loop!
    # (Make sure the compute_global_maxs function from the previous step is in this file)
    channel_maxs = compute_global_maxs(train_loader, device)

    # ------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------
    model = build_model(
        input_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES
    ).to(device)
    weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            channel_maxs,
            config.USE_MEL,
            epoch,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            channel_maxs,
            config.USE_MEL,
        )

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

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
