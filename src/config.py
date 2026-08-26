from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TIER1_DATA = DATA_DIR / "processed" / "tier1_modeling.csv"
TIER2_DATA = DATA_DIR / "processed" / "tier2_modeling.csv"

TIER1_MODEL = MODEL_DIR / "tier1_model.joblib"
TIER2_MODEL = MODEL_DIR / "tier2_model.joblib"

TRAIN_END = "2023-12-31"
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"

RIDGE_ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]
