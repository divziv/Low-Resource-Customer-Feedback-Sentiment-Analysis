
import copy
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


from src.config import (
    MODELS,
    get_model_paths,
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE,
    NUM_EPOCHS,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    EARLY_STOPPING_PATIENCE,
    MAX_GRAD_NORM,
    DEVICE,
    RANDOM_SEED,
)


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



def train_model(model_name):


    print("\n========================")
    print("Training:", model_name)
    print("========================")


    output_dir, model_dir, figure_dir = get_model_paths(
        model_name
    )


    # Load tokenizer and datasets for this model

    (
        train_dataset,
        valid_dataset,
        _,
        _,
        tokenizer
    ) = get_datasets(
        model_name
    )


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



    model, tokenizer = load_model(
        model_name
    )



    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )



    total_steps = (
        len(train_loader)
        *
        NUM_EPOCHS
        //
        GRADIENT_ACCUMULATION
    )



    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            WARMUP_RATIO * total_steps
        ),
        num_training_steps=total_steps,
    )



    class_weights = torch.tensor(
        [77/23, 1.0],
        dtype=torch.float
    ).to(DEVICE)



    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )



    best_accuracy = -1
    best_model = None
    best_metrics = None
    patience = 0


    train_losses = []
    val_scores = []



    for epoch in range(NUM_EPOCHS):


        print(
            f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
        )


        model.train()

        running_loss = 0


        optimizer.zero_grad()



        for step, batch in enumerate(
            tqdm(train_loader)
        ):


            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            labels = batch["labels"].to(DEVICE)



            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            loss = loss_function(
                outputs.logits,
                labels
            )


            loss = loss / GRADIENT_ACCUMULATION


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



            running_loss += loss.item()



        train_loss = (
            running_loss
            /
            len(train_loader)
        )



        model.eval()


        predictions = []
        labels_all = []



        with torch.no_grad():


            for batch in valid_loader:


                input_ids = batch["input_ids"].to(DEVICE)

                attention_mask = batch["attention_mask"].to(DEVICE)

                labels = batch["labels"].to(DEVICE)



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


                labels_all.extend(
                    labels.cpu().numpy()
                )



        metrics = compute_metrics(
            predictions,
            labels_all
        )


        accuracy = metrics["accuracy"]



        print("Loss:", train_loss)

        print("Accuracy:", accuracy)



        train_losses.append(
            train_loss
        )

        val_scores.append(
            accuracy
        )



        if accuracy > best_accuracy:


            best_accuracy = accuracy

            best_metrics = metrics


            best_model = copy.deepcopy(
                model
            )


            patience = 0



            save_model(
                best_model,
                tokenizer,
                model_dir
            )


        else:


            patience += 1


            if patience >= EARLY_STOPPING_PATIENCE:

                print(
                    "Early stopping"
                )

                break



    save_metrics(
        best_metrics,
        output_dir
    )


    print_metrics(
        best_metrics
    )


    plot_training_loss(
        train_losses,
        figure_dir
    )


    plot_validation_accuracy(
        val_scores,
        figure_dir
    )




for model_name in MODELS:

    train_model(
        model_name
    )
