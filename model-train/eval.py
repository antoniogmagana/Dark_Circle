import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import config
from models import MODEL_REGISTRY
from train import db_worker_init
from dataset import VehicleDataset
from preprocess import preprocess_for_training


def run_evaluation():
    device = config.DEVICE
    print(f"Using device: {device}")

    # 1. LOAD METADATA AND NORMALIZATION STATS
    # No hardcoded fallbacks allowed. If train.py hasn't generated this, we halt.
    if not os.path.exists(config.META_SAVE_PATH):
        raise FileNotFoundError(
            f"Metadata file {config.META_SAVE_PATH} not found. "
            "Please run train.py first so it can compute and save the dynamic channel_maxs."
        )

    meta = torch.load(config.META_SAVE_PATH, map_location=device)
    channel_maxs = meta["channel_maxs"]
    print(f"Loaded normalization stats from metadata: {channel_maxs.tolist()}")

    # 2. Initialize the Test Dataset
    test_ds = VehicleDataset(split="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=db_worker_init,
        pin_memory=True,
        persistent_workers=True,
    )

    # 3. Build and Load Model
    model_cls = MODEL_REGISTRY[config.MODEL_NAME]
    model = model_cls(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        use_mel=config.USE_MEL,
    ).to(device)

    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print(f"Successfully loaded {config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError(
            f"Could not find {config.MODEL_SAVE_PATH}. Did training finish?"
        )

    model.eval()

    all_preds = []
    all_labels = []

    print(f"Starting Evaluation on {len(test_ds)} samples...")

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            # Apply the SAME preprocessing used in training
            x = preprocess_for_training(x, channel_maxs, use_mel=config.USE_MEL)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"Eval Progress: Batch {i+1}/{len(test_loader)}")

    # 4. Metrics Calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    # Get class names dynamically based on the active training mode
    if config.TRAINING_MODE == "detection":
        class_names = ["background", "vehicle"]
    elif config.TRAINING_MODE == "category":
        class_names = [config.CLASS_MAP[i] for i in sorted(config.CLASS_MAP.keys())]
    elif config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in config.INSTANCE_TO_CLASS.items()}
        class_names = [inv_map[i] for i in range(config.NUM_CLASSES)]
    else:
        class_names = [str(i) for i in range(config.NUM_CLASSES)]

    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)

    # Console output
    print("\n" + "=" * 60)
    print(f"{'Vehicle Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(
            f"{name:<25} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {f1[i]:<10.4f}"
        )

    print("-" * 60)
    print(f"Overall Test Accuracy: {accuracy:.4f}")
    print("=" * 60 + "\n")

    # 5. Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {config.MODEL_NAME}")
    plt.savefig(config.IMG_SAVE_PATH)
    print(f"Confusion matrix saved to {config.IMG_SAVE_PATH}")


if __name__ == "__main__":
    run_evaluation()
