
from pathlib import Path
import torch


# Models to train

MODELS = [
    "microsoft/deberta-v3-base",
    "roberta-base",
    "bert-base-uncased",
    "google/electra-base-discriminator",
    "microsoft/deberta-v3-small",
    "microsoft/deberta-v3-large",
    "microsoft/MiniLM-L12-H384-uncased",
]


NUM_LABELS = 2
MAX_LENGTH = 256



# Default model
# Change this when running one model manually

MODEL_NAME = MODELS[0]



# Data paths

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "train.csv"
VALID_FILE = DATA_DIR / "valid.csv"
TEST_FILE = DATA_DIR / "test.csv"
UNLABELLED_FILE = DATA_DIR / "unlabelled.csv"



# Model folders

def get_model_paths(model_name):

    model_folder = model_name.replace("/", "_")

    output_dir = (
        PROJECT_ROOT
        /
        "outputs"
        /
        model_folder
    )

    model_dir = output_dir / "best_model"

    figure_dir = output_dir / "figures"


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    return output_dir, model_dir, figure_dir



# Current model paths

OUTPUT_DIR, MODEL_DIR, FIGURE_DIR = get_model_paths(
    MODEL_NAME
)



# Training

RANDOM_SEED = 42


TRAIN_BATCH_SIZE = 4

VALID_BATCH_SIZE = 16


LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01


NUM_EPOCHS = 5


GRADIENT_ACCUMULATION = 2


WARMUP_RATIO = 0.10


EARLY_STOPPING_PATIENCE = 2


MAX_GRAD_NORM = 1.0



# Device

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# Mixed precision

FP16 = False



# Labels

LABEL2ID = {
    "negative": 0,
    "positive": 1,
}


ID2LABEL = {
    0: "negative",
    1: "positive",
}



# Output files

METRICS_FILE = OUTPUT_DIR / "metrics.txt"

PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"

TRAINING_CURVE = OUTPUT_DIR / "training_loss.png"

VALIDATION_CURVE = OUTPUT_DIR / "validation_accuracy.png"
