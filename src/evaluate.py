
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    MODELS,
    get_model_paths,
    VALID_BATCH_SIZE,
    DEVICE,
)

from src.dataset import get_datasets
from src.model import load_saved_model
from src.utils import compute_metrics, print_metrics



def evaluate_model(model_name):

    print("Evaluating:", model_name)


    output_dir, model_dir, figure_dir = get_model_paths(
        model_name
    )


    # Load dataset with correct tokenizer

    (
        _,
        _,
        test_dataset,
        _,
        _
    ) = get_datasets(
        model_name
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False
    )


    model, tokenizer = load_saved_model(
        model_dir
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
                attention_mask=attention_mask
            )


            preds = torch.argmax(
                outputs.logits,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )


            labels.extend(
                batch_labels.cpu().numpy()
            )


    metrics = compute_metrics(
        predictions,
        labels
    )


    print_metrics(
        metrics
    )


    with open(
        output_dir / "test_metrics.txt",
        "w"
    ) as file:

        for key, value in metrics.items():

            file.write(
                f"{key}: {value:.4f}\n"
            )


    print(
        "Saved:",
        output_dir / "test_metrics.txt"
    )



for model_name in MODELS:

    evaluate_model(
        model_name
    )
