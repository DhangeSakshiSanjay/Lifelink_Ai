from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ML_DIR = PROJECT_ROOT / "ml"
ARTIFACT_DIR = ML_DIR / "artifacts"
MODEL_DIR = ML_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "kidney_dataset.csv"

TARGET_COLUMN = "Match_Status"

RANDOM_STATE = 42
TEST_SIZE = 0.20