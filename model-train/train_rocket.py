import os
import torch
import numpy as np
import joblib
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

import config
from dataset import VehicleDataset
from models import MODEL_REGISTRY
from train import db_worker_init, compute_global_maxs
from preprocess import preprocess_for_training


def gather_data_into_ram(loader, device, channel_maxs):
    """
    Rapidly iterates through the PyTorch DataLoader, applies GPU preprocessing,
    and stacks the entire dataset into a NumPy array for scikit-learn.
    """
    X_all = []
    y_all = []

    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)

            # CRITICAL: use_mel=False. MiniRocket needs the 1D raw waveforms!
            x = preprocess_for_training(x, channel_maxs, use_mel=False)

            X_all.append(x.cpu().numpy())
            y_all.append(y.numpy())

            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"Data Extraction Progress: Batch {i+1}/{len(loader)}")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def main():
    device = config.DEVICE
    print(f"Using device: {device} for preprocessing.")

    # 1. Force MiniRocket Configurations
    config.MODEL_NAME = "ClassificationMiniRocket"
    config.USE_MEL = False

    # --- NEW: Create Directory and Save Config Snapshot ---
    print(f"Starting MiniRocket Run ID: {config.RUN_ID}")
    config.save_config_snapshot()

    # 2. Datasets & Loaders
    train_ds = VehicleDataset(split="train")
    test_ds = VehicleDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # Order doesn't matter for whole-batch fitting
        num_workers=config.NUM_WORKERS,
        worker_init_fn=db_worker_init,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=db_worker_init,
    )

    # 3. Calculate Scaling Metrics
    channel_maxs = compute_global_maxs(train_loader, device)

    # Save metadata so eval.py could theoretically use it later
    torch.save(
        {
            "channel_maxs": channel_maxs,
            "model_name": config.MODEL_NAME,
            "use_mel": False,
        },
        config.META_SAVE_PATH,
    )

    # 4. Extract Data into RAM
    print("\n--- Extracting Training Data ---")
    X_train, y_train = gather_data_into_ram(train_loader, device, channel_maxs)

    print("\n--- Extracting Test Data ---")
    X_test, y_test = gather_data_into_ram(test_loader, device, channel_maxs)

    # 5. Fit MiniRocket
    print(f"\nTraining ClassificationMiniRocket on shape {X_train.shape}...")
    model = MODEL_REGISTRY["ClassificationMiniRocket"]()
    model.fit(X_train, y_train)

    # Scikit-learn models must be saved with joblib, not torch.save
    save_path = config.MODEL_SAVE_PATH.replace(".pth", ".joblib")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"MiniRocket saved to {save_path}")

    # 6. Immediate Evaluation
    print("\nEvaluating MiniRocket Test Set...")
    preds = model.predict(X_test)

    # Determine class names dynamically
    if config.TRAINING_MODE == "detection":
        class_names = ["background", "vehicle"]
    elif config.TRAINING_MODE == "category":
        class_names = [config.CLASS_MAP[i] for i in sorted(config.CLASS_MAP.keys())]
    elif config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in config.INSTANCE_TO_CLASS.items()}
        class_names = [inv_map[i] for i in range(config.NUM_CLASSES)]
    else:
        class_names = [str(i) for i in range(config.NUM_CLASSES)]

    # Metrics Calculation
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average=None)
    recall = recall_score(y_test, preds, average=None)
    f1 = f1_score(y_test, preds, average=None)
    cm = confusion_matrix(y_test, preds)

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

    # Save Confusion Matrix
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
    plt.title(f"Confusion Matrix: ClassificationMiniRocket")
    plt.savefig(config.IMG_SAVE_PATH.replace(".png", "_minirocket.png"))
    print("Confusion matrix saved!")


if __name__ == "__main__":
    main()
