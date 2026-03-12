import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, 
    matthews_corrcoef, 
    roc_auc_score,
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score
)
import warnings

# Suppress the sklearn warning for undefined metrics when only 1 class is present
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import config
from dataset import VehicleDataset, db_worker_init
from models import build_model
from preprocess import preprocess_for_training


def evaluate_directory(run_dir_path):
    # 1. Check for existing artifacts
    report_path = run_dir_path / "evaluation_report.txt"
    if report_path.exists():
        print(f"Skipping: {run_dir_path} (Artifacts already exist)")
        return
        
    print(f"\nEvaluating: {run_dir_path}")
    
    # 2. Reconstruct context from path
    # Assuming path is saved_models/<mode>/<model_name>/<run_id>
    parts = run_dir_path.parts
    training_mode = parts[-3]
    model_name = parts[-2]
    
    # Override config dynamically for this specific evaluation run
    config.TRAINING_MODE = training_mode
    config.MODEL_NAME = model_name
    
    # Re-trigger the dynamic class weighting/sizing logic based on the overridden mode
    if training_mode == "detection":
        config.NUM_CLASSES = 2
    elif training_mode == "category":
        config.NUM_CLASSES = len(config.CLASS_MAP)
    elif training_mode == "instance":
        config.NUM_CLASSES = len(config.INSTANCE_TO_CLASS)
        
    # 3. Load Metadata
    meta_path = run_dir_path / "meta.pt"
    if not meta_path.exists():
        print(f"  [!] Missing meta.pt in {run_dir_path}")
        return
        
    meta = torch.load(meta_path, map_location=config.DEVICE, weights_only=False)
    sigma = meta["sigma"].to(config.DEVICE)
    epsilon = meta["epsilon"]
    
    # 4. Build Dataset & DataLoader
    test_ds = VehicleDataset(split="test", config=config)
    
    if len(test_ds) == 0:
        print(f"  [!] No test samples found for {training_mode}.")
        return
        
    custom_worker_init = partial(db_worker_init, config=config)
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        worker_init_fn=custom_worker_init
    )
    
    # 5. Load Model
    model = build_model(
        input_channels=config.IN_CHANNELS, 
        num_classes=config.NUM_CLASSES, 
        config=config
    ).to(config.DEVICE)
    
    model_path = run_dir_path / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    model.eval()
    
    # 6. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.perf_counter()
    
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(config.DEVICE)
            x = preprocess_for_training(x, sigma, epsilon, config=config)
            
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    end_time = time.perf_counter()
    
    # 7. Calculate Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    latency_ms = ((end_time - start_time) / len(test_ds)) * 1000
    
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # New Metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Safe ROC-AUC (Fixes the UndefinedMetricWarning spam)
    unique_classes = len(np.unique(all_labels))
    if unique_classes > 1:
        try:
            if config.NUM_CLASSES == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')
        
    # Confusion Matrix
    target_labels = list(range(config.NUM_CLASSES))
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)
    
    # Safe FAR
    far = None
    if training_mode == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
    # Safe Per-Class Accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_acc[np.isnan(per_class_acc)] = 0.0 
        
    # 8. Save Artifacts
    with open(report_path, "w") as f:
        # Keep exact formatting for aggregate_results.py parsing
        f.write(f"Run Directory: {run_dir_path.name}\n")
        f.write(f"Mode: {training_mode} | Model: {model_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Latency: {latency_ms:.4f} ms/sample\n")
        
        if far is not None:
            f.write(f"False Alarm Rate: {far * 100:.3f}%\n")
            
        f.write("\nPer-Class Accuracy:\n")
        if training_mode == "detection":
            f.write(f"  Background (0): {per_class_acc[0]:.4f}\n")
            f.write(f"  Target (1): {per_class_acc[1]:.4f}\n")
        elif training_mode == "category":
            for k, v in config.CLASS_MAP.items():
                if k < len(per_class_acc):
                    f.write(f"  {v} ({k}): {per_class_acc[k]:.4f}\n")
        
    # Plot Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name} ({training_mode})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(run_dir_path / "conf_matrix.png")
    plt.close()
    
    print(f"  [+] Generated report and confusion matrix in {run_dir_path}")


def main():
    base_dir = Path("saved_models")
    if not base_dir.exists():
        print("No saved_models directory found.")
        return
        
    run_dirs = [p.parent for p in base_dir.rglob("best_model.pth")]
    
    for run_dir in run_dirs:
        evaluate_directory(run_dir)


if __name__ == "__main__":
    main()