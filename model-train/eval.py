import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
import models
import preprocess
from train import VehicleStreamer


def run_evaluation():
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    model = models.ClassificationCNN(
        in_channels=checkpoint["in_channels"], num_classes=len(config.CLASS_MAP)
    ).to(config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        VehicleStreamer(split="test"),
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
    )

    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = preprocess.align_and_upsample(
                preprocess.zero_center_window(
                    features.to(config.DEVICE, non_blocking=True)
                )
            )

            if isinstance(model, (models.ClassificationCNN, models.DetectionCNN)):
                features = preprocess.extract_mel_spectrogram(features)

            _, predicted = torch.max(model(features).data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

            if i >= config.EVAL_STEPS - 1:
                break

    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    print("\n" + "=" * 50)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 50)

    target_names = [config.CLASS_MAP[i] for i in range(len(precision))]
    for i, name in enumerate(target_names):
        print(f"{name:<15} | {precision[i]:<10.4f} | {recall[i]:<10.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Multi-Modal Vehicle Classification Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    run_evaluation()
