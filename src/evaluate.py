import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import *
from src.dataset import get_datasets
from src.model import load_saved_model
from src.utils import (
    compute_metrics,
    plot_confusion_matrix,
    save_metrics,
    print_metrics,
)


def evaluate():
    print("Loading model...")

    model, tokenizer = load_saved_model()

    _, _, test_dataset, _, _ = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
    )

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in tqdm(test_loader):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            batch_labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    metrics = compute_metrics(predictions, labels)

    print_metrics(metrics)

    save_metrics(metrics)

    plot_confusion_matrix(labels, predictions)

    prediction_df = pd.DataFrame(
        {
            "True Label": labels,
            "Predicted Label": predictions,
        }
    )

    prediction_df["True Label"] = prediction_df["True Label"].map(ID2LABEL)
    prediction_df["Predicted Label"] = prediction_df["Predicted Label"].map(ID2LABEL)

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX}")


if __name__ == "__main__":
    evaluate()