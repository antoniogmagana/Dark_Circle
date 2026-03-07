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
from train import VehicleStreamer
from preprocess import (
    preprocess_for_training,
    preprocess_window,
    extract_mel_spectrogram,
)


def build_model_for_eval(checkpoint):
    model_cls = MODEL_REGISTRY[config.MODEL_NAME]
    use_mel = checkpoint.get("use_mel", config.USE_MEL)

    model = model_cls(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        use_mel=use_mel,
    ).to(config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, use_mel


def get_class_names():
    if config.TRAINING_MODE == "detection":
        return ["background", "vehicle"]

    if config.TRAINING_MODE == "category":
        return [config.CLASS_MAP[i] for i in sorted(config.CLASS_MAP.keys())]

    if config.TRAINING_MODE == "instance":
        inv = {v: k for k, v in config.INSTANCE_TO_CLASS.items()}
        return [inv[i] for i in range(len(inv))]

    raise ValueError(f"Unknown TRAINING_MODE: {config.TRAINING_MODE}")


def run_evaluation():
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    model, use_mel = build_model_for_eval(checkpoint)

    test_loader = DataLoader(
        VehicleStreamer(split="test"),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            if config.BATCH_MODE:
                x = preprocess_for_training(x, use_mel=use_mel)
            else:
                x = preprocess_window(x[0]).unsqueeze(0)
                if use_mel:
                    x = extract_mel_spectrogram(x)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if i >= config.EVAL_STEPS - 1:
                break

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    class_names = get_class_names()
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # ---------------------------------------------------------
    # CSV EXPORT
    # ---------------------------------------------------------
    # Ensure directory exists
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)

    metrics_path = os.path.join(
        config.EVAL_RESULTS_DIR, f"metrics_{config.MODEL_NAME}_{timestamp}.csv"
    )

    confusion_path = os.path.join(
        config.EVAL_RESULTS_DIR, f"confusion_{config.MODEL_NAME}_{timestamp}.csv"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_df = pd.DataFrame(
        {
            "class_id": list(range(len(class_names))),
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    overall_df = pd.DataFrame(
        {
            "metric": ["accuracy", "macro_precision", "macro_recall", "macro_f1"],
            "value": [
                accuracy,
                precision.mean(),
                recall.mean(),
                f1.mean(),
            ],
        }
    )

    metrics_path = f"eval_metrics_{config.MODEL_NAME}_{timestamp}.csv"
    confusion_path = f"eval_confusion_{config.MODEL_NAME}_{timestamp}.csv"

    metrics_df.to_csv(metrics_path, index=False)
    overall_df.to_csv(metrics_path, mode="a", index=False)
    pd.DataFrame(cm).to_csv(confusion_path, index=False)

    print(f"\nSaved metrics → {metrics_path}")
    print(f"Saved confusion matrix → {confusion_path}")

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 60)

    for i, name in enumerate(class_names):
        print(
            f"{name:<25} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {f1[i]:<10.4f}"
        )

    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------
    # Confusion matrix plot
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False if config.TRAINING_MODE == "instance" else True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix ({config.TRAINING_MODE} mode)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evaluation()
