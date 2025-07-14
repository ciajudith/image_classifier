from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR  = BASE_DIR / "data" / "animals10"
MODEL_DIR = BASE_DIR / "models"

IMG_HEIGHT = 150
IMG_WIDTH  = 150
BATCH_SIZE = 32
EPOCHS     = 10
