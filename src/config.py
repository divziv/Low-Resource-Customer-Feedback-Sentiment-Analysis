
from pathlib import Path
import torch


# Model

MODEL_NAME = "microsoft/deberta-v3-base"

NUM_LABELS = 2
MAX_LENGTH = 256


# Paths

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"

MODEL_FOLDER_NAME = MODEL_NAME.replace("/", "_")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_FOLDER_NAME

MODEL_DIR = OUTPUT_DIR / "best_model"

FIGURE_DIR = OUTPUT_DIR / "figures"


TRAIN_FILE = DATA_DIR / "train.csv"
VALID_FILE = DATA_DIR / "valid.csv"
TEST_FILE = DATA_DIR / "test.csv"
UNLABELLED_FILE = DATA_DIR / "unlabelled.csv"


# Create folders

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Training

RANDOM_SEED = 42


# Batch sizes

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 16


# Lower learning rate for DeBERTa stability

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01


# Train longer with early stopping

NUM_EPOCHS = 5


# Gradient accumulation

GRADIENT_ACCUMULATION = 2


# Scheduler

WARMUP_RATIO = 0.10


# Stop if validation does not improve

EARLY_STOPPING_PATIENCE = 2


# Gradient clipping

MAX_GRAD_NORM = 1.0


# Device

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Mixed precision training

FP16 = False


# Output files

MODEL_SAVE_PATH = MODEL_DIR

METRICS_FILE = OUTPUT_DIR / "metrics.txt"

PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"

TRAINING_CURVE = OUTPUT_DIR / "training_loss.png"

VALIDATION_CURVE = OUTPUT_DIR / "validation_accuracy.png"


# Labels

LABEL2ID = {
    "negative": 0,
    "positive": 1,
}


ID2LABEL = {
    0: "negative",
    1: "positive",
}
