import os
import time
import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader
from scipy.special import softmax
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

import config
from dataset import VehicleDataset
from train import db_worker_init
from preprocess import preprocess_for_training

def gather_test_data(loader, device, mu, sigma, epsilon):
    """Extracts only the test set for final evaluation."""
    X_all, y_all = [], []
    print(f"Extracting test features for {len(loader.dataset)} samples...")
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            x = preprocess_for_training(x, mu, sigma, epsilon, use_mel=False)
            X_all.append(x.cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

def main():
    device = config.DEVICE
    print(f"Using device for preprocessing: {device}")

    if not os.path.exists(config.META_SAVE_PATH):
        raise FileNotFoundError(f"Metadata {config.META_SAVE_PATH} not found. Please run train_rocket.py first.")

    meta = torch.load(config.META_SAVE_PATH, map_location=device)
    mu, sigma, epsilon = meta["mu"], meta["sigma"], meta["epsilon"]
    print(f"Loaded normalization stats from metadata: {meta}")

    # 1. Initialize Test Dataset ONLY (Aligned with eval.py)
    test_ds = VehicleDataset(split="test")
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS, 
        worker_init_fn=db_worker_init,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 2. Load Model
    save_path = config.MODEL_SAVE_PATH.replace(".pth", ".joblib")
    if os.path.exists(save_path):
        model = joblib.load(save_path)
        print(f"Successfully loaded {save_path}")
    else:
        raise FileNotFoundError(f"Could not find {save_path}.")

    # 3. Extract Test Data into RAM
    X_test, y_test = gather_test_data(test_loader, device, mu, sigma, epsilon)

    # 4. Evaluate
    print("Running MiniRocket Prediction on Test Set...")
    start_time = time.perf_counter()
    
    # We explicitly extract the features here to get both predictions and ROC-AUC scores
    X_test_feat = model.transformer.transform(X_test)
    scores = model.classifier.decision_function(X_test_feat)
    all_preds = model.classifier.predict(X_test_feat)
    
    end_time = time.perf_counter()

    accuracy = accuracy_score(y_test, all_preds)
    mcc = matthews_corrcoef(y_test, all_preds)
    
    total_samples = len(y_test)
    latency_ms = ((end_time - start_time) / total_samples) * 1000

    # --- FIX: Convert raw decision scores to probabilities for multiclass ---
    if config.NUM_CLASSES == 2:
        auc = roc_auc_score(y_test, scores)
    else:
        prob_scores = softmax(scores, axis=1)
        auc = roc_auc_score(y_test, prob_scores, multi_class='ovr')

    cm = confusion_matrix(y_test, all_preds)
    if config.TRAINING_MODE == "detection" and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        far = None

    if config.TRAINING_MODE == "detection":
        class_names = ["background", "vehicle"]
    elif config.TRAINING_MODE == "category":
        class_names = [config.CLASS_MAP[i] for i in sorted(config.CLASS_MAP.keys())]
    elif config.TRAINING_MODE == "instance":
        inv_map = {v: k for k, v in config.INSTANCE_TO_CLASS.items()}
        class_names = [inv_map[i] for i in range(config.NUM_CLASSES)]
    else:
        class_names = [str(i) for i in range(config.NUM_CLASSES)]

    precision = precision_score(y_test, all_preds, average=None, zero_division=0)
    recall = recall_score(y_test, all_preds, average=None, zero_division=0)
    f1 = f1_score(y_test, all_preds, average=None, zero_division=0)

    print("\n" + "=" * 60)
    print(f"{'Vehicle Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"{name:<25} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {f1[i]:<10.4f}")
    print("-" * 60)
    print(f"Overall MiniRocket Test Accuracy: {accuracy:.4f}")
    print(f"MCC: {mcc:.4f} | AUC: {auc:.4f} | Latency: {latency_ms:.2f} ms/sample")
    if far is not None:
        print(f"False Alarm Rate: {far:.4%}")
    print("=" * 60 + "\n")

    # Generate Evaluation Report File for the Aggregator
    report_path = os.path.join(config.RUN_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("MODEL PERFORMANCE REPORT\n")
        f.write(f"Timestamp: {config.RUN_ID}\n")
        f.write(f"Mode: {config.TRAINING_MODE} | Model: ClassificationMiniRocket\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"MCC:       {mcc:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n")
        f.write(f"Latency:   {latency_ms:.4f} ms/sample\n")
        if far is not None:
            f.write(f"False Alarm Rate: {far:.4%}\n")

    # Plot Test Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Test Confusion Matrix: MiniRocket ({config.TRAINING_MODE})")
    test_img_path = config.IMG_SAVE_PATH.replace(".png", "_test_rocket.png")
    plt.savefig(test_img_path)
    print(f"Test confusion matrix saved to {test_img_path}")

if __name__ == "__main__":
    main()