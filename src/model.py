
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    MODEL_NAME,
    NUM_LABELS,
    MODEL_DIR,
    DEVICE,
    LABEL2ID,
    ID2LABEL,
)



def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # Always train in FP32 for stability
    model = model.float()

    model.to(DEVICE)

    print(
        "Base model dtype:",
        next(model.parameters()).dtype
    )

    return model, tokenizer





def save_model(model, tokenizer):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )

    print(
        f"Model saved to: {MODEL_DIR}"
    )





def load_saved_model():

    """
    Load the fine-tuned model from outputs/best_model
    for evaluation or inference.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )


    # Ensure FP32 evaluation
    model = model.float()

    model.to(DEVICE)


    print(
        "Loaded saved model dtype:",
        next(model.parameters()).dtype
    )


    return model, tokenizer
