import os
import csv
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

# New imports for advanced metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from data_generator import augment_batch
from dataset import VehicleDataset, db_worker_init
from models import build_model
from preprocess import preprocess_for_training
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
            current_snr_range = getattr(config, "AUGMENT_SNR_RANGE", (10, 30))
            x = augment_batch(x, snr_range=current_snr_range)
        # -----------------------------------------------------------------

        # Preprocessing on the GPU
        x = preprocess_for_training(x, sigma, epsilon, config=config)        

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, sigma, epsilon, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Preprocessing on the GPU (Injecting config)
            x = preprocess_for_training(x, sigma, epsilon, config=config)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    total_samples = len(all_labels)
    avg_loss = total_loss / total_samples
    
    # Calculate global metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    # Calculate per-class accuracy safely using the confusion matrix
    target_labels = list(range(config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        # Handle NaN values for classes that might not appear in this specific validation batch
        per_class_acc[np.isnan(per_class_acc)] = 0.0 

    return avg_loss, accuracy, precision, recall, f1, per_class_acc


def compute_stats(calib_loader, config):
    channels_sq_sum = torch.zeros(config.IN_CHANNELS)
    total_samples = 0
    
    with torch.no_grad():
        for x, _ in calib_loader:
            window_mean = x.mean(dim=-1, keepdim=True)
            x_ac = x - window_mean
            channels_sq_sum += torch.sum(x_ac**2, dim=[0, 2])
            total_samples += x.shape[0] * x.shape[2]
            
    sigma = torch.sqrt(channels_sq_sum / total_samples)
    epsilon = 1e-8
    return sigma, epsilon


def compute_noise_floor(calib_loader):
    all_stds = []
    
    with torch.no_grad():
        for x, _ in calib_loader:
            window_stds = torch.std(x, dim=2)
            all_stds.append(window_stds)
            
    all_stds = torch.cat(all_stds, dim=0)
    noise_floor = torch.quantile(all_stds, q=0.05, dim=0)
    return noise_floor


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

    sigma, epsilon = compute_stats(calib_loader, config=config)
    noise_floor = compute_noise_floor(calib_loader)

    train_ds.noise_floor = noise_floor

    print(f"Computed Std: {sigma}")
    print(f"Computed Noise Floor: {noise_floor}")

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
            x_dummy = preprocess_for_training(x_dummy, sigma, epsilon, config=config)
            if hasattr(model, 'fit_extractor'):
                model.fit_extractor(x_dummy)
            model(x_dummy)
            break

    if len(config.CLASS_WEIGHTS) == config.NUM_CLASSES:
        weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print(f"Warning: CLASS_WEIGHTS len ({len(config.CLASS_WEIGHTS)}) != NUM_CLASSES ({config.NUM_CLASSES}). Defaulting to unweighted loss.")
        criterion = nn.CrossEntropyLoss()

    optimizer = model.get_optimizer()

    # ------------------------------------------------------------
    # CSV Initialization & Dynamic Headers
    # ------------------------------------------------------------
    if config.TRAINING_MODE == "detection":
        class_names = ["Val_Acc_background", "Val_Acc_vehicle"]
    elif config.TRAINING_MODE == "category":
        class_names = [f"Val_Acc_{config.CLASS_MAP[i]}" for i in sorted(config.CLASS_MAP.keys())]
    elif config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in config.INSTANCE_TO_CLASS.items()}
        class_names = [f"Val_Acc_{inv_map[i]}" for i in range(config.NUM_CLASSES)]
    else:
        class_names = [f"Val_Acc_Class_{i}" for i in range(config.NUM_CLASSES)]

    headers = [
        "Epoch", "Train_Loss", "Train_Acc", 
        "Val_Loss", "Val_Acc", "Val_Precision", "Val_Recall", "Val_F1"
    ] + class_names

    with open(config.METRICS_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    # ------------------------------------------------------------
    # Training loop with Dynamic "Best Model" criteria
    # ------------------------------------------------------------
    # Default to tracking accuracy if the config variable is missing
    target_metric_name = getattr(config, "BEST_MODEL_METRIC", "val_acc")
    
    # If we are tracking loss, we want the LOWEST number. 
    # For everything else (acc, f1, precision), we want the HIGHEST number.
    if target_metric_name == "val_loss":
        best_metric_value = float('inf') 
    else:
        best_metric_value = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, sigma, epsilon, config, epoch
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, per_class_acc = evaluate(
            model, val_loader, criterion, device, sigma, epsilon, config
        )

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

        # Convert the numpy array of per-class accuracies to formatted strings
        class_values = [f"{per_class_acc[i]:.4f}" for i in range(config.NUM_CLASSES)]

        with open(config.METRICS_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            row_data = [
                epoch, 
                f"{train_loss:.4f}", f"{train_acc:.4f}", 
                f"{val_loss:.4f}", f"{val_acc:.4f}",
                f"{val_prec:.4f}", f"{val_rec:.4f}", f"{val_f1:.4f}"
            ] + class_values
            writer.writerow(row_data)

        # Map the string name from config to the actual variable we just calculated
        metrics_dict = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec
        }
        
        current_metric_value = metrics_dict.get(target_metric_name, val_acc)

        # Determine if this epoch is the new "best"
        is_best = False
        if target_metric_name == "val_loss":
            if current_metric_value < best_metric_value:
                is_best = True
        else:
            if current_metric_value > best_metric_value:
                is_best = True

        # Ensure we only save the model weights when our target metric hits a new peak
        if is_best:
            best_metric_value = current_metric_value
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  --> New Best Model saved with {target_metric_name}: {best_metric_value:.4f}")

    print(f"\nTraining Complete. Best {target_metric_name} Achieved: {best_metric_value:.4f}")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()