import csv
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from data_generator import augment_batch
from dataset import VehicleDataset
from models import build_model
from preprocess import preprocess_for_training
import config


class EpochShuffleSampler(Sampler):
    """Shuffles dataset indices with a distinct seed each epoch."""

    def __init__(self, dataset, base_seed, num_epochs):
        self.dataset = dataset
        self.seeds = [base_seed + i for i in range(num_epochs)]
        self.epoch = 0

    def set_epoch(self, epoch):
        """Call before each epoch (0-indexed)."""
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seeds[self.epoch])
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def train_one_epoch(model, loader, optimizer, criterion, device, config, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y, dataset_names) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # -----------------------------------------------------------------
        # DYNAMIC SYNTHESIS: SNR Augmentation (Controlled via config)
        # -----------------------------------------------------------------
        if getattr(config, "AUGMENT_SNR", False):
            current_snr_range = getattr(config, "AUGMENT_SNR_RANGE", (10, 30))
            x = augment_batch(x, snr_range=current_snr_range)
        # -----------------------------------------------------------------

        # Preprocessing on the GPU
        x = preprocess_for_training(x, config=config)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for x, y, dataset_names in loader:
            x, y = x.to(device), y.to(device)

            # Preprocessing on the GPU
            x = preprocess_for_training(x, config=config)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    total_samples = len(all_labels)
    avg_loss = total_loss / total_samples

    # Calculate global metrics
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    # Calculate per-class accuracy safely using the confusion matrix
    target_labels = list(range(config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    return avg_loss, accuracy, precision, recall, f1, per_class_acc


def main():
    device = config.DEVICE
    print("Using device:", device)

    print(f"Starting Run ID: {config.RUN_ID}")
    print(f"Saving to: {config.RUN_DIR}")
    config.save_config_snapshot()

    # ------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------
    train_ds = VehicleDataset(split="train", config=config)
    val_ds = VehicleDataset(split="val", config=config)
    epoch_sampler = EpochShuffleSampler(
        train_ds, base_seed=config.INSTANCE_SEED, num_epochs=config.EPOCHS
    )

    print(f"Total training samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=epoch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # ------------------------------------------------------------
    # Save Metadata (Now storing dicts)
    # ------------------------------------------------------------
    torch.save(
        {
            "model_name": config.MODEL_NAME,
            "use_mel": config.USE_MEL,
        },
        config.META_SAVE_PATH,
    )

    print(f"Saved model metadata to: {config.META_SAVE_PATH}")

    # ------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------
    model = build_model(
        input_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        config=config,
    ).to(device)

    print("Performing dummy pass to initialize Lazy modules...")
    model.eval()
    with torch.no_grad():
        for x_dummy, _, ds_names in train_loader:
            x_dummy = x_dummy.to(device)

            x_dummy = preprocess_for_training(x_dummy, config=config)

            if hasattr(model, "fit_extractor"):
                model.fit_extractor(x_dummy[:32])
                model(x_dummy[:32])
            else:
                model(x_dummy)
            break

    if len(config.CLASS_WEIGHTS) == config.NUM_CLASSES:
        weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print(
            f"Warning: CLASS_WEIGHTS len ({len(config.CLASS_WEIGHTS)}) "
            f"!= NUM_CLASSES ({config.NUM_CLASSES}). "
            f"Defaulting to unweighted loss."
        )
        criterion = nn.CrossEntropyLoss()

    optimizer = model.get_optimizer()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=(
            "min"
            if getattr(config, "BEST_MODEL_METRIC", "val_acc") == "val_loss"
            else "max"
        ),
        factor=0.5,
        patience=3,
    )

    # ------------------------------------------------------------
    # CSV Initialization & Dynamic Headers
    # ------------------------------------------------------------
    if config.TRAINING_MODE == "detection":
        class_names = ["Val_Acc_background", "Val_Acc_vehicle"]
    elif config.TRAINING_MODE == "category":
        class_names = [
            f"Val_Acc_{config.CLASS_MAP[i]}" for i in sorted(config.CLASS_MAP.keys())
        ]
    elif config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in config.INSTANCE_TO_CLASS.items()}
        class_names = [f"Val_Acc_{inv_map[i]}" for i in range(config.NUM_CLASSES)]
    else:
        class_names = [f"Val_Acc_Class_{i}" for i in range(config.NUM_CLASSES)]

    headers = [
        "Epoch",
        "Train_Loss",
        "Train_Acc",
        "Val_Loss",
        "Val_Acc",
        "Val_Precision",
        "Val_Recall",
        "Val_F1",
    ] + class_names

    with open(config.METRICS_LOG_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    # ------------------------------------------------------------
    # Training loop with Dynamic "Best Model" criteria
    # ------------------------------------------------------------
    target_metric_name = getattr(config, "BEST_MODEL_METRIC", "val_acc")

    if target_metric_name == "val_loss":
        best_metric_value = float("inf")
    else:
        best_metric_value = 0.0
    best_val_f1 = 0.0

    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 8

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        epoch_sampler.set_epoch(epoch - 1)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, per_class_acc = evaluate(
            model, val_loader, criterion, device, config
        )

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

        class_values = [f"{per_class_acc[i]:.4f}" for i in range(config.NUM_CLASSES)]

        with open(config.METRICS_LOG_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            row_data = [
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{val_prec:.4f}",
                f"{val_rec:.4f}",
                f"{val_f1:.4f}",
            ] + class_values
            writer.writerow(row_data)

        metrics_dict = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
        }

        current_metric_value = metrics_dict.get(target_metric_name, val_acc)

        is_best = False
        if target_metric_name == "val_loss":
            if current_metric_value < best_metric_value:
                is_best = True
        else:
            if current_metric_value > best_metric_value:
                is_best = True

        if is_best:
            best_metric_value = current_metric_value
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(
                f"  --> New Best Model saved with "
                f"{target_metric_name}: {best_metric_value:.4f}"
            )
        else:
            epochs_no_improve += 1
            print(
                f"  --> No improvement: "
                f"{epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs."
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

        scheduler.step(current_metric_value)

    # Update meta with best val_f1 for use in fused evaluation weighting
    meta = torch.load(config.META_SAVE_PATH, weights_only=False)
    meta["val_f1"] = best_val_f1
    torch.save(meta, config.META_SAVE_PATH)

    print(
        f"\nTraining Complete. Best {target_metric_name} Achieved: "
        f"{best_metric_value:.4f}"
    )
    print(f"Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
