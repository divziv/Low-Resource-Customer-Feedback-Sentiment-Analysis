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


# Save model

def save_model(model, tokenizer):
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)


# Load saved model

def load_saved_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH
    )

    model.to(DEVICE)

    return model, tokenizer


# Count trainable parameters

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# Freeze encoder layers

def freeze_encoder(model):
    for parameter in model.deberta.parameters():
        parameter.requires_grad = False

    return model


# Unfreeze encoder layers

def unfreeze_encoder(model):
    for parameter in model.deberta.parameters():
        parameter.requires_grad = True

    return model


# Print model summary

def print_model_summary(model):
    print("\nModel Summary")
    print("-" * 50)

    print(f"Model Name       : {MODEL_NAME}")
    print(f"Device           : {DEVICE}")
    print(f"Number of Labels : {NUM_LABELS}")
    print(f"Max Length       : {MAX_LENGTH}")
    print(f"Trainable Params : {count_parameters(model):,}")


if __name__ == "__main__":
    tokenizer = load_tokenizer()

    model = load_model()

    print_model_summary(model)

    print("\nTokenizer Vocabulary Size:")
    print(tokenizer.vocab_size)

    print("\nClassification Labels:")
    print(model.config.id2label)