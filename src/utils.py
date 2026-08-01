import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import *


# Set random seed

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Compute evaluation metrics

def compute_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Save metrics to file

def save_metrics(metrics):
    with open(METRICS_FILE, "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.4f}\n")


# Plot confusion matrix

def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"],
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX)
    plt.close()


# Plot training loss

def plot_training_loss(losses):
    plt.figure(figsize=(8, 5))

    plt.plot(losses, marker="o")

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(TRAINING_CURVE)
    plt.close()


# Plot validation accuracy

def plot_validation_accuracy(scores):
    plt.figure(figsize=(8, 5))

    plt.plot(scores, marker="o")

    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(VALIDATION_CURVE)
    plt.close()


# Print metrics nicely

def print_metrics(metrics):
    print("\nEvaluation Results")
    print("-" * 30)

    for key, value in metrics.items():
        print(f"{key.capitalize():12}: {value:.4f}")