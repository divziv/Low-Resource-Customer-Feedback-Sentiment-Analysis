import json
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import *


# Load tokenizer

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


# Load model

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.to(DEVICE)

    return model


# Get device information

def get_device_info():

    if torch.cuda.is_available():

        return {
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_GB": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9,
                2
            ),
        }

    return {
        "cuda_available": False,
        "gpu_name": "CPU",
        "gpu_memory_GB": 0,
    }


# Count trainable parameters

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# Save model

def save_model(model, tokenizer):

    MODEL_SAVE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        MODEL_SAVE_PATH
    )

    tokenizer.save_pretrained(
        MODEL_SAVE_PATH
    )


    model_info = {

        "model_name": MODEL_NAME,

        "architecture": model.__class__.__name__,

        "num_labels": NUM_LABELS,

        "labels": ID2LABEL,

        "max_length": MAX_LENGTH,

        "trainable_parameters": count_parameters(model),

        "device": str(DEVICE),

        **get_device_info(),
    }


    with open(
        MODEL_SAVE_PATH / "model_info.json",
        "w"
    ) as file:

        json.dump(
            model_info,
            file,
            indent=4
        )


    print(
        f"Model saved to: {MODEL_SAVE_PATH}"
    )


# Load saved model

def load_saved_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SAVE_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH
    )

    model.to(DEVICE)

    return model, tokenizer



# Freeze encoder layers

def freeze_encoder(model):

    if hasattr(model, "distilbert"):

        encoder = model.distilbert

    elif hasattr(model, "deberta"):

        encoder = model.deberta

    elif hasattr(model, "roberta"):

        encoder = model.roberta

    elif hasattr(model, "bert"):

        encoder = model.bert

    else:

        raise AttributeError(
            "Unsupported model architecture."
        )


    for parameter in encoder.parameters():

        parameter.requires_grad = False


    return model



# Unfreeze encoder layers

def unfreeze_encoder(model):

    if hasattr(model, "distilbert"):

        encoder = model.distilbert

    elif hasattr(model, "deberta"):

        encoder = model.deberta

    elif hasattr(model, "roberta"):

        encoder = model.roberta

    elif hasattr(model, "bert"):

        encoder = model.bert

    else:

        raise AttributeError(
            "Unsupported model architecture."
        )


    for parameter in encoder.parameters():

        parameter.requires_grad = True


    return model



# Print model summary

def print_model_summary(model):

    print("\nModel Summary")
    print("-" * 50)

    print(f"Model Name       : {MODEL_NAME}")

    print(f"Architecture     : {model.__class__.__name__}")

    print(f"Device           : {DEVICE}")

    print(f"Number of Labels : {NUM_LABELS}")

    print(f"Max Length       : {MAX_LENGTH}")

    print(
        f"Trainable Params : {count_parameters(model):,}"
    )


    if torch.cuda.is_available():

        print(
            f"GPU              : {torch.cuda.get_device_name(0)}"
        )



if __name__ == "__main__":

    tokenizer = load_tokenizer()

    model = load_model()

    print_model_summary(model)

    print("\nTokenizer Vocabulary Size:")
    print(tokenizer.vocab_size)

    print("\nClassification Labels:")
    print(model.config.id2label)