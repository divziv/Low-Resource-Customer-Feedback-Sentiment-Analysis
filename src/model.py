
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    NUM_LABELS,
    DEVICE,
    LABEL2ID,
    ID2LABEL,
)



def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        attn_implementation="eager",
    )


    # Train using FP32
    model = model.float()

    model.to(DEVICE)


    print(
        "Loaded model:",
        model_name
    )

    print(
        "Model dtype:",
        next(model.parameters()).dtype
    )


    return model, tokenizer





def save_model(
    model,
    tokenizer,
    model_dir
):

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model.save_pretrained(
        model_dir
    )


    tokenizer.save_pretrained(
        model_dir
    )


    print(
        f"Model saved to: {model_dir}"
    )






def load_saved_model(model_dir):

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        attn_implementation="eager",
    )


    model = model.float()

    model.to(DEVICE)


    print(
        "Loaded saved model:",
        model_dir
    )


    print(
        "Model dtype:",
        next(model.parameters()).dtype
    )


    return model, tokenizer
