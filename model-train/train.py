import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import VehicleDataset
from models import build_model
from preprocess import preprocess_for_training, extract_mel_spectrogram
import config


def train_one_epoch(model, loader, optimizer, criterion, device, use_mel, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # Preprocessing
        if config.BATCH_MODE:
            x = preprocess_for_training(x, use_mel=use_mel)
        else:
            x = preprocess_for_training(x, use_mel=use_mel)

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
                x = preprocess_for_training(x, use_mel=use_mel)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def compute_sensor_stats(dataset):
    mins = {s: float("inf") for s in config.TRAIN_SENSORS}
    maxs = {s: float("-inf") for s in config.TRAIN_SENSORS}

    for i in range(len(dataset)):
        ds, inst, sec = dataset.index[i]
        tables = dataset.instance_to_tables[inst]

        for sensor in config.TRAIN_SENSORS:
            matches = [t for t in tables if f"_{sensor}_" in t]
            if not matches:
                continue

            table = matches[0]
            sr_native = config.NATIVE_SR[ds][sensor]
            raw = fetch_sensor_batch(dataset.cursor, table, sr_native, sec)
            if not raw:
                continue

            if sensor == "accel":
                arr = torch.tensor(raw, dtype=torch.float32).T
            else:
                arr = torch.tensor([r[0] for r in raw], dtype=torch.float32)[None, :]

            mins[sensor] = min(mins[sensor], arr.min().item())
            maxs[sensor] = max(maxs[sensor], arr.max().item())

    return mins, maxs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------
    train_ds = VehicleDataset(split="train")
    sensor_mins, sensor_maxs = compute_sensor_stats(train_ds)
    train_ds.set_normalization(sensor_mins, sensor_maxs)

    val_ds = VehicleDataset(split="val")
    val_ds.set_normalization(sensor_mins, sensor_maxs)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------
    model = build_model(
        input_channels=config.NUM_CHANNELS, num_classes=config.NUM_CLASSES
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    use_mel = config.USE_MEL

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_mel, epoch
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_mel)

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------
    torch.save(model.state_dict(), "final_model.pt")
    print("Training complete. Model saved to final_model.pt")


if __name__ == "__main__":
    main()
