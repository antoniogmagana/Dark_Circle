"""
Batch trainer — loads data ONCE per (sensor, mode), trains all models on it.

Usage:
    # Train all models on seismic detection (1 data load, 6 model trains):
    TRAIN_SENSOR=seismic TRAINING_MODE=detection \
        poetry run python train_batch.py ResNet1D WaveformClassificationCNN ClassificationLSTM

    # Train all models (default):
    TRAIN_SENSOR=seismic TRAINING_MODE=detection \
        poetry run python train_batch.py

Called by run_pipeline.sh for efficient sweeps.
"""

import os
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data_generator import augment_batch
from dataset import VehicleDataset, MemoryDataset, db_worker_init, preload_to_memory
from models import build_model, MODEL_REGISTRY
from preprocess import preprocess
import config


# =====================================================================
# Reuse training helpers from train.py
# =====================================================================

class EarlyStopping:
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


def train_one_epoch(model, loader, optimizer, criterion, device, cfg, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for x, y, _ds in loader:
        x, y = x.to(device), y.to(device)
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

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_samples += bs

    return running_loss / running_samples, running_correct / running_samples


@torch.inference_mode()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y, _ds in loader:
        x, y = x.to(device), y.to(device)
        x = preprocess(x, config=cfg)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
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

    return running_loss / n, accuracy, precision, recall, f1, per_class_acc


def compute_noise_floor(calib_loader):
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
# Config builder for a specific model (reuses shared config values)
# =====================================================================

# Import the hyperparameter table directly from config.py
from config import _HYPERPARAMS, SHAPE_MAP


def build_model_config(model_name, base_config):
    """
    Build a config namespace for a specific model, inheriting all shared
    values from the base config and overlaying model-specific hyperparams.
    """
    # Start with a copy of all base config values
    cfg_dict = {}
    for key in dir(base_config):
        if key.isupper() and not key.startswith("_"):
            cfg_dict[key] = getattr(base_config, key)

    # Override model-specific fields
    cfg_dict["MODEL_NAME"] = model_name
    cfg_dict["USE_MEL"] = SHAPE_MAP.get(model_name, "1D") == "2D"

    # Apply sensor-specific hyperparameters for this model
    hp_key = (model_name, cfg_dict["TRAIN_SENSOR"])
    if hp_key in _HYPERPARAMS:
        cfg_dict.update(_HYPERPARAMS[hp_key])

    # Generate unique run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dict["RUN_ID"] = run_id
    cfg_dict["RUN_DIR"] = os.path.join(
        "saved_models", cfg_dict["TRAINING_MODE"],
        cfg_dict["TRAIN_SENSOR"], model_name, run_id
    )
    cfg_dict["MODEL_SAVE_PATH"] = os.path.join(cfg_dict["RUN_DIR"], "best_model.pth")
    cfg_dict["META_SAVE_PATH"] = os.path.join(cfg_dict["RUN_DIR"], "meta.pt")
    cfg_dict["JSON_LOG_PATH"] = os.path.join(cfg_dict["RUN_DIR"], "hyperparameters.json")
    cfg_dict["METRICS_LOG_PATH"] = os.path.join(cfg_dict["RUN_DIR"], "metrics.csv")

    return SimpleNamespace(**cfg_dict)


def save_config_snapshot(cfg):
    """Save config to JSON (standalone version for SimpleNamespace)."""
    os.makedirs(cfg.RUN_DIR, exist_ok=True)
    config_dict = {}
    for key in dir(cfg):
        if key.isupper() and not key.startswith("_"):
            value = getattr(cfg, key)
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            elif isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            elif isinstance(value, (int, float, str, list, dict, bool, tuple, type(None))):
                config_dict[key] = value
    with open(cfg.JSON_LOG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)


# =====================================================================
# Train one model on pre-loaded data
# =====================================================================

def train_model(model_name, train_loader, val_loader, noise_floors, base_config):
    """Train a single model using already-loaded data."""
    device = base_config.DEVICE

    cfg = build_model_config(model_name, base_config)
    save_config_snapshot(cfg)

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}  (USE_MEL={cfg.USE_MEL})")
    print(f"  Run:   {cfg.RUN_DIR}")
    print(f"{'='*60}")

    # Save metadata
    torch.save(
        {"model_name": model_name, "sensor": cfg.TRAIN_SENSOR,
         "use_mel": cfg.USE_MEL, "noise_floors": noise_floors},
        cfg.META_SAVE_PATH,
    )

    # Build model
    model = build_model(
        input_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        config=cfg,
    ).to(device)

    # Dummy pass for lazy modules / MiniRocket
    needs_init = hasattr(model, "fit_extractor") or any(
        isinstance(m, nn.LazyLinear) for m in model.modules()
    )
    if needs_init:
        model.eval()
        with torch.no_grad():
            for x_dummy, _, _ in train_loader:
                x_dummy = x_dummy.to(device)
                x_dummy = preprocess(x_dummy, config=cfg)
                if hasattr(model, "fit_extractor"):
                    model.fit_extractor(x_dummy)
                model(x_dummy)
                break

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    if len(cfg.CLASS_WEIGHTS) == cfg.NUM_CLASSES:
        weights = torch.tensor(cfg.CLASS_WEIGHTS, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = model.get_optimizer()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    target_metric = getattr(cfg, "BEST_MODEL_METRIC", "val_acc")
    es_mode = "min" if target_metric == "val_loss" else "max"
    early_stopping = EarlyStopping(
        patience=getattr(cfg, "EARLY_STOP_PATIENCE", 8), mode=es_mode,
    )
    best_value = float("inf") if es_mode == "min" else 0.0
    grad_clip = getattr(cfg, "GRAD_CLIP", 1.0)

    # CSV
    with open(cfg.METRICS_LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "Epoch", "LR", "Train_Loss", "Train_Acc",
            "Val_Loss", "Val_Acc", "Val_Precision", "Val_Recall", "Val_F1",
        ])

    # Training loop
    for epoch in range(1, cfg.EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, grad_clip,
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, per_class_acc = evaluate(
            model, val_loader, criterion, device, cfg,
        )

        print(
            f"  Epoch {epoch:>2}/{cfg.EPOCHS}  lr={lr:.1e}  "
            f"TrL={train_loss:.4f} TrA={train_acc:.4f}  "
            f"VL={val_loss:.4f} VA={val_acc:.4f} VF1={val_f1:.4f}",
            flush=True,
        )

        old_lr = lr
        scheduler.step(val_loss if target_metric == "val_loss" else val_f1)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"    --> LR reduced: {old_lr:.2e} → {new_lr:.2e}")

        with open(cfg.METRICS_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}",
                f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}",
                f"{val_prec:.4f}", f"{val_rec:.4f}", f"{val_f1:.4f}",
            ])

        metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                    "val_precision": val_prec, "val_recall": val_rec}
        current = metrics.get(target_metric, val_acc)

        is_best = current < best_value if es_mode == "min" else current > best_value
        if is_best:
            best_value = current
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"    --> Best model ({target_metric}: {best_value:.4f})")

        if early_stopping(current):
            print(f"    Early stopping at epoch {epoch}.")
            break

    print(f"  Done. Best {target_metric}: {best_value:.4f}\n")

    # Free GPU memory before next model
    del model, optimizer, criterion, scheduler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return best_value


# =====================================================================
# Main
# =====================================================================

def main():
    device = config.DEVICE

    # Parse model list from CLI, default to all models
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = list(MODEL_REGISTRY.keys())

    # Validate
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"Unknown model: {m}")
            print(f"Available: {list(MODEL_REGISTRY.keys())}")
            return

    print(f"Device:  {device}")
    print(f"Sensor:  {config.TRAIN_SENSOR}")
    print(f"Mode:    {config.TRAINING_MODE}")
    print(f"Signal:  {config.REF_SAMPLE_RATE} Hz × {config.SAMPLE_SECONDS}s")
    print(f"Models:  {', '.join(models)}")
    print()

    # ------------------------------------------------------------------
    # Load data ONCE
    # ------------------------------------------------------------------
    print("Loading datasets from database...")
    train_ds = VehicleDataset(split="train", config=config)
    val_ds = VehicleDataset(split="val", config=config)

    print(f"Training samples: {len(train_ds):,}")
    print(f"Validation samples: {len(val_ds):,}")

    custom_worker_init = partial(db_worker_init, config=config)

    # Noise floor
    calib_size = max(1, int(len(train_ds) * 0.10))
    calib_indices = torch.randperm(len(train_ds))[:calib_size].tolist()
    calib_ds = torch.utils.data.Subset(train_ds, calib_indices)
    calib_loader = DataLoader(
        calib_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, worker_init_fn=custom_worker_init,
    )

    print("Computing noise floor...")
    noise_floors = compute_noise_floor(calib_loader)
    train_ds.noise_floors = noise_floors
    val_ds.noise_floors = noise_floors
    print(f"Noise floors: {noise_floors}")

    # Cache in memory / GPU
    if getattr(config, "CACHE_SAMPLES", False):
        print("\nPre-loading training data...")
        train_ds = preload_to_memory(train_ds, config)
        print("Pre-loading validation data...")
        val_ds = preload_to_memory(val_ds, config)

        loader_kwargs = dict(
            batch_size=config.BATCH_SIZE,
            num_workers=0,
        )
    else:
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
    # Train each model sequentially (reusing loaded data)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DATA LOADED — TRAINING {len(models)} MODELS")
    print(f"{'='*60}")

    results = {}
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]", end="")

        import time
        time.sleep(1)  # Guarantee unique RUN_IDs

        best = train_model(model_name, train_loader, val_loader, noise_floors, config)
        results[model_name] = best

    # Summary
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {config.TRAIN_SENSOR} / {config.TRAINING_MODE}")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {config.BEST_MODEL_METRIC}")
    print(f"  {'-'*45}")
    for model_name, best in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {model_name:<30} {best:.4f}")
    print()


if __name__ == "__main__":
    main()