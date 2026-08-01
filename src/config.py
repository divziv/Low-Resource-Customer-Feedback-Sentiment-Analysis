"""
Configuration file for Low-Resource Customer Feedback Sentiment Analysis
"""

from pathlib import Path
import torch

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "best_model"
FIGURE_DIR = PROJECT_ROOT / "figures"

TRAIN_FILE = DATA_DIR / "train.csv"
VALID_FILE = DATA_DIR / "valid.csv"
TEST_FILE = DATA_DIR / "test.csv"
UNLABELLED_FILE = DATA_DIR / "unlabelled.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Model

MODEL_NAME = "microsoft/deberta-v3-base"

NUM_LABELS = 2
MAX_LENGTH = 256

# Training

RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 3

GRADIENT_ACCUMULATION = 2
WARMUP_RATIO = 0.10

EARLY_STOPPING_PATIENCE = 2

# Device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16 = torch.cuda.is_available()

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