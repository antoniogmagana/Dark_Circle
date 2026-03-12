import os
import time
import json
from pathlib import Path
from types import SimpleNamespace
from functools import partial

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
    matthews_corrcoef,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Notice: global 'config' is no longer imported
from models import MODEL_REGISTRY
from dataset import VehicleDataset, db_worker_init
from preprocess import preprocess_for_training


def evaluate_single_run(run_dir: Path):
    """Runs evaluation for a specific model directory using its local config."""
    
    # 1. LOAD LOCAL CONFIGURATION
    config_path = run_dir / "hyperparameters.json"
    if not config_path.exists():
        print(f"  [!] Skipping: Config file {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    # Convert dict to a SimpleNamespace so we can use dot notation (e.g., run_config.BATCH_SIZE)
    # Standard lists of floats/ints loaded from JSON will remain perfectly intact here.
    run_config = SimpleNamespace(**config_dict)
    
    # Re-cast device from string back to torch.device.
    device = torch.device(run_config.DEVICE)
    
    # Define dynamic paths based on the run directory
    model_path = run_dir / "best_model.pth"
    meta_path = run_dir / "meta.pt"
    report_path = run_dir / "evaluation_report.txt"
    img_path = run_dir / "conf_matrix.png"

    # 2. LOAD METADATA AND NORMALIZATION STATS
    if not meta_path.exists():
        print(f"  [!] Skipping: Metadata file {meta_path} not found.")
        return

    meta = torch.load(meta_path, map_location=device)
    sigma, epsilon = meta["sigma"], meta["epsilon"]

    # 3. INITIALIZE THE TEST DATASET (Dependency Injection)
    test_ds = VehicleDataset(split="test", config=run_config)
    
    # Inject the config into the worker initialization function
    custom_worker_init = partial(db_worker_init, config=run_config)

    test_loader = DataLoader(
        test_ds,
        batch_size=run_config.BATCH_SIZE,
        shuffle=False,
        num_workers=run_config.NUM_WORKERS,
        worker_init_fn=custom_worker_init,
        pin_memory=True,
        persistent_workers=True,
    )

    # 4. BUILD AND LOAD MODEL
    model_cls = MODEL_REGISTRY[run_config.MODEL_NAME]
    model = model_cls(
        in_channels=run_config.IN_CHANNELS,
        num_classes=run_config.NUM_CLASSES,
        config=run_config,
        use_mel=run_config.USE_MEL,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    # --- START TIMING ---
    start_time = time.perf_counter()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Pass the loaded config to the preprocessor
            x = preprocess_for_training(x, sigma, epsilon, config=run_config)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- END TIMING ---
    end_time = time.perf_counter()

    # 5. METRICS CALCULATION
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    latency_ms = ((end_time - start_time) / len(all_labels)) * 1000

    if run_config.NUM_CLASSES == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', labels=list(range(run_config.NUM_CLASSES)))

    cm = confusion_matrix(all_labels, all_preds)
    far = None
    if run_config.TRAINING_MODE == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Handle class name mapping safely if JSON cast integer dictionary keys to strings
    if run_config.TRAINING_MODE == "detection":
        class_names = ["background", "vehicle"]
    elif run_config.TRAINING_MODE == "category":
        class_map = {int(k) if str(k).isdigit() else k: v for k, v in run_config.CLASS_MAP.items()}
        class_names = [class_map[i] for i in sorted(class_map.keys())]
    elif run_config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in run_config.INSTANCE_TO_CLASS.items()}
        class_names = [inv_map[i] for i in range(run_config.NUM_CLASSES)]
    else:
        class_names = [str(i) for i in range(run_config.NUM_CLASSES)]

    target_labels = list(range(run_config.NUM_CLASSES))
    precision = precision_score(all_labels, all_preds, labels=target_labels, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, labels=target_labels, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, labels=target_labels, average=None, zero_division=0)

    # 6. SAVE REPORT TO TEXT FILE
    with open(report_path, "w") as f:
        f.write("MODEL PERFORMANCE REPORT\n")
        f.write(f"Run Directory: {run_dir.name}\n")
        f.write(f"Mode: {run_config.TRAINING_MODE} | Model: {run_config.MODEL_NAME}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"MCC:       {mcc:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n")
        f.write(f"Latency:   {latency_ms:.4f} ms/sample\n")
        if far is not None:
            f.write(f"False Alarm Rate: {far:.4%}\n")

    # 7. PLOT CONFUSION MATRIX
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {run_config.MODEL_NAME}")
    plt.savefig(img_path)
    plt.close() # CRITICAL for batch processing memory management
    
    print(f"  [+] Generated report and confusion matrix in {run_dir}")


def scan_and_evaluate(base_dir="./saved_models"):
    """Scans the directory structure and evaluates runs missing artifacts."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Base directory {base_path} does not exist.")
        return

    print(f"Scanning {base_path} for evaluation targets...")
    
    for model_path in base_path.rglob("best_model.pth"):
        run_dir = model_path.parent
        report_path = run_dir / "evaluation_report.txt"
        img_path = run_dir / "conf_matrix.png"
        
        if not report_path.exists() or not img_path.exists():
            print(f"\nEvaluating: {run_dir}")
            try:
                evaluate_single_run(run_dir)
            except Exception as e:
                print(f"  [!] Error evaluating {run_dir}: {e}")
        else:
            print(f"Skipping: {run_dir} (Artifacts already exist)")


if __name__ == "__main__":
    scan_and_evaluate()