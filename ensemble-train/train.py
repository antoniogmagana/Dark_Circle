import os
import csv
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data_generator import augment_batch
from dataset import VehicleDataset, db_worker_init
from models import build_model
from preprocess import preprocess
import config


# =====================================================================
# Helpers
# =====================================================================

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience=5, mode="max", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def __call__(self, value):
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def build_class_headers(cfg):
    """Generate per-class CSV column names based on training mode."""
    if cfg.TRAINING_MODE == "detection":
        return ["Val_Acc_background", "Val_Acc_vehicle"]
    elif cfg.TRAINING_MODE == "category":
        return [f"Val_Acc_{cfg.CLASS_MAP[i]}" for i in sorted(cfg.CLASS_MAP.keys())]
    elif cfg.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in cfg.INSTANCE_TO_CLASS.items()}
        return [f"Val_Acc_{inv_map[i]}" for i in range(cfg.NUM_CLASSES)]
    return [f"Val_Acc_Class_{i}" for i in range(cfg.NUM_CLASSES)]


# =====================================================================
# Training & Evaluation Loops
# =====================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, cfg, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for x, y, _dataset_names in loader:
        x, y = x.to(device), y.to(device)

        # Optional SNR augmentation (before normalization)
        if getattr(cfg, "AUGMENT_SNR", False):
            x = augment_batch(x, snr_range=getattr(cfg, "AUGMENT_SNR_RANGE", (10, 30)))

        x = preprocess(x, config=cfg)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_samples += batch_size

    return running_loss / running_samples, running_correct / running_samples


@torch.inference_mode()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y, _dataset_names in loader:
        x, y = x.to(device), y.to(device)
        x = preprocess(x, config=cfg)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    avg_loss = running_loss / n

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    accuracy = (preds_arr == labels_arr).mean()
    precision = precision_score(labels_arr, preds_arr, average="weighted", zero_division=0)
    recall = recall_score(labels_arr, preds_arr, average="weighted", zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)

    cm = confusion_matrix(labels_arr, preds_arr, labels=list(range(cfg.NUM_CLASSES)))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0

    return avg_loss, accuracy, precision, recall, f1, per_class_acc


# =====================================================================
# Calibration (noise floor only — used for synthetic background generation)
# =====================================================================

def compute_noise_floor(calib_loader):
    """5th-percentile noise floor per dataset (for synthetic background generation)."""
    all_stds = {}

    with torch.no_grad():
        for x, _, dataset_names in calib_loader:
            window_stds = torch.std(x, dim=2)
            for i, ds in enumerate(dataset_names):
                if ds not in all_stds:
                    all_stds[ds] = []
                all_stds[ds].append(window_stds[i].unsqueeze(0))

    return {
        ds: torch.quantile(torch.cat(stds, dim=0), q=0.05, dim=0)
        for ds, stds in all_stds.items()
    }


# =====================================================================
# Main
# =====================================================================

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Starting Run ID: {config.RUN_ID}")
    print(f"Saving to: {config.RUN_DIR}")
    config.save_config_snapshot()

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_ds = VehicleDataset(split="train", config=config)
    val_ds = VehicleDataset(split="val", config=config)
    custom_worker_init = partial(db_worker_init, config=config)

    print(f"Total training samples: {len(train_ds)}")

    # Calibration subset (10%) — only needed for noise floor estimation
    calib_size = max(1, int(len(train_ds) * 0.10))
    calib_indices = torch.randperm(len(train_ds))[:calib_size].tolist()
    calib_ds = torch.utils.data.Subset(train_ds, calib_indices)

    calib_loader = DataLoader(
        calib_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
    )

    print(f"Estimating noise floor from a {calib_size}-sample calibration subset...")
    noise_floors = compute_noise_floor(calib_loader)
    train_ds.noise_floors = noise_floors

    print(f"Per-Dataset Noise Floors: {noise_floors}")

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ------------------------------------------------------------------
    # Save metadata (noise floors for synthetic generation only)
    # ------------------------------------------------------------------
    torch.save(
        {
            "model_name": config.MODEL_NAME,
            "use_mel": config.USE_MEL,
            "noise_floors": noise_floors,
        },
        config.META_SAVE_PATH,
    )
    print(f"Saved metadata to: {config.META_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model(
        input_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        config=config,
    ).to(device)

    # Dummy forward pass to initialise LazyLinear dimensions
    print("Performing dummy pass to initialize Lazy modules...")
    model.eval()
    with torch.no_grad():
        for x_dummy, _, _ds_names in train_loader:
            x_dummy = x_dummy.to(device)
            x_dummy = preprocess(x_dummy, config=config)
            if hasattr(model, "fit_extractor"):
                model.fit_extractor(x_dummy)
            model(x_dummy)
            break

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    if len(config.CLASS_WEIGHTS) == config.NUM_CLASSES:
        weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print(
            f"Warning: CLASS_WEIGHTS len ({len(config.CLASS_WEIGHTS)}) "
            f"!= NUM_CLASSES ({config.NUM_CLASSES}). Using unweighted loss."
        )
        criterion = nn.CrossEntropyLoss()

    optimizer = model.get_optimizer()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # ------------------------------------------------------------------
    # Early stopping setup
    # ------------------------------------------------------------------
    target_metric_name = getattr(config, "BEST_MODEL_METRIC", "val_acc")
    es_mode = "min" if target_metric_name == "val_loss" else "max"
    early_stopping = EarlyStopping(
        patience=getattr(config, "EARLY_STOP_PATIENCE", 8),
        mode=es_mode,
    )

    best_metric_value = float("inf") if es_mode == "min" else 0.0

    # ------------------------------------------------------------------
    # CSV logger
    # ------------------------------------------------------------------
    class_headers = build_class_headers(config)
    csv_headers = [
        "Epoch", "LR",
        "Train_Loss", "Train_Acc",
        "Val_Loss", "Val_Acc", "Val_Precision", "Val_Recall", "Val_F1",
    ] + class_headers

    with open(config.METRICS_LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(csv_headers)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    grad_clip = getattr(config, "GRAD_CLIP", 1.0)

    for epoch in range(1, config.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.EPOCHS}  (lr={current_lr:.2e})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, config, grad_clip=grad_clip,
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, per_class_acc = evaluate(
            model, val_loader, criterion, device, config,
        )

        print(
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}",
            flush=True,
        )

        # Step the scheduler on the primary validation metric
        scheduler_metric = val_loss if target_metric_name == "val_loss" else val_f1
        scheduler.step(scheduler_metric)

        # Log to CSV
        class_values = [f"{per_class_acc[i]:.4f}" for i in range(config.NUM_CLASSES)]
        with open(config.METRICS_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{current_lr:.2e}",
                 f"{train_loss:.4f}", f"{train_acc:.4f}",
                 f"{val_loss:.4f}", f"{val_acc:.4f}",
                 f"{val_prec:.4f}", f"{val_rec:.4f}", f"{val_f1:.4f}"]
                + class_values
            )

        # Best model checkpoint
        metrics_dict = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
        }
        current_value = metrics_dict.get(target_metric_name, val_acc)

        is_best = (
            current_value < best_metric_value
            if es_mode == "min"
            else current_value > best_metric_value
        )
        if is_best:
            best_metric_value = current_value
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  --> Best model saved ({target_metric_name}: {best_metric_value:.4f})")

        # Early stopping check
        if early_stopping(current_value):
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining complete. Best {target_metric_name}: {best_metric_value:.4f}")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
