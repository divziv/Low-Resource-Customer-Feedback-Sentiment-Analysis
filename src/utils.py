
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import RANDOM_SEED



def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def compute_metrics(labels, predictions):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }




def save_metrics(metrics, output_dir):

    if metrics is None:
        print("No metrics to save")
        return


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    file_path = output_dir / "metrics.txt"


    with open(file_path, "w") as file:

        for key, value in metrics.items():

            file.write(
                f"{key}: {value:.4f}\n"
            )


    print(
        f"Metrics saved to: {file_path}"
    )




def plot_training_loss(loss_history, figure_path):

    plt.figure(figsize=(6,4))

    plt.plot(
        loss_history,
        marker="o"
    )

    plt.title(
        "Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid()

    plt.savefig(
        figure_path,
        bbox_inches="tight"
    )

    plt.close()




def plot_validation_accuracy(acc_history, figure_path):

    plt.figure(figsize=(6,4))

    plt.plot(
        acc_history,
        marker="o"
    )

    plt.title(
        "Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.grid()

    plt.savefig(
        figure_path,
        bbox_inches="tight"
    )

    plt.close()

def print_metrics(metrics):

    print("\nEvaluation Results")
    print("-" * 30)

    for key, value in metrics.items():

        print(
            f"{key.capitalize():12}: {value:.4f}"
        )
