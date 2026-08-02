
import copy
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import *
from src.dataset import get_datasets
from src.model import load_model, save_model

from src.utils import (
    set_seed,
    compute_metrics,
    plot_training_loss,
    plot_validation_accuracy,
    save_metrics,
    print_metrics,
)


set_seed(RANDOM_SEED)


train_dataset, valid_dataset, _, _, tokenizer = get_datasets()


train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)


valid_loader = DataLoader(
    valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
)



# Load model

model = load_model()

print(
    "Model dtype:",
    next(model.parameters()).dtype
)



# Optimizer

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)



# Scheduler

total_training_steps = (
    len(train_loader)
    * NUM_EPOCHS
    //
    GRADIENT_ACCUMULATION
)


warmup_steps = int(
    WARMUP_RATIO * total_training_steps
)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)



# Class imbalance handling

class_weights = torch.tensor(
    [77/23, 1.0],
    dtype=torch.float32,
    device=DEVICE
)



loss_function = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)



training_losses = []
validation_scores = []


best_accuracy = -1.0
best_metrics = None
best_model = None

patience = 0


MAX_GRAD_NORM = 1.0




def train_one_epoch():

    model.train()

    running_loss = 0.0

    optimizer.zero_grad()



    for step, batch in enumerate(
        tqdm(train_loader)
    ):


        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)



        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )



        # Convert logits to float32
        # prevents FP16 loss error

        loss = loss_function(
            outputs.logits.float(),
            labels
        )



        if torch.isnan(loss) or torch.isinf(loss):

            print(
                "Invalid loss detected. Skipping batch."
            )

            optimizer.zero_grad()

            continue



        loss = (
            loss /
            GRADIENT_ACCUMULATION
        )



        loss.backward()



        if (
            step + 1
        ) % GRADIENT_ACCUMULATION == 0:



            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )



            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()



        running_loss += (
            loss.item()
            *
            GRADIENT_ACCUMULATION
        )




    # Handle leftover gradients

    if len(train_loader) % GRADIENT_ACCUMULATION != 0:


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )


        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()



    return (
        running_loss /
        len(train_loader)
    )






def validate():

    model.eval()


    predictions = []

    labels_list = []



    with torch.no_grad():


        for batch in valid_loader:


            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            labels = batch["labels"].to(DEVICE)



            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )



            preds = torch.argmax(
                outputs.logits,
                dim=1
            )



            predictions.extend(
                preds.cpu().numpy()
            )


            labels_list.extend(
                labels.cpu().numpy()
            )



    print(
        "Prediction distribution:",
        torch.bincount(
            torch.tensor(predictions)
        )
    )


    print(
        "Label distribution:",
        torch.bincount(
            torch.tensor(labels_list)
        )
    )



    return compute_metrics(
        predictions,
        labels_list
    )







for epoch in range(NUM_EPOCHS):


    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )



    train_loss = train_one_epoch()


    metrics = validate()


    accuracy = metrics["accuracy"]



    training_losses.append(
        train_loss
    )


    validation_scores.append(
        accuracy
    )



    print(
        f"Training Loss : {train_loss:.4f}"
    )


    print(
        f"Validation Accuracy : {accuracy:.4f}"
    )





    if accuracy > best_accuracy:


        best_accuracy = accuracy

        best_metrics = metrics

        patience = 0



        best_model = copy.deepcopy(
            model
        )



        save_model(
            best_model,
            tokenizer
        )



        print(
            "Best model saved."
        )



    else:


        patience += 1



        if patience >= EARLY_STOPPING_PATIENCE:


            print(
                "Early stopping."
            )

            break







print(
    "\nTraining completed."
)




plot_training_loss(
    training_losses
)



plot_validation_accuracy(
    validation_scores
)




if best_metrics is not None:


    save_metrics(
        best_metrics
    )


    print_metrics(
        best_metrics
    )


else:


    print(
        "No valid model was saved."
    )
