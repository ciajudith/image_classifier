from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "user_data"
MODEL_DIR = BASE_DIR / "models"

# On passe à 224×224 pour coller à MobileNetV2
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Nombre total d’époques (Phase1+Phase2)
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5
EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
