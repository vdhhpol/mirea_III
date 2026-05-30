import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "configs" / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv(BASE_DIR / ".env")

class Config:
    PREPROCESSOR_PATH = BASE_DIR / os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "artifacts/lightgbm_tuned.pkl")
    CALIBRATOR_PATH = BASE_DIR / os.getenv("CALIBRATOR_PATH", "artifacts/calibrator.pkl")
    THRESHOLD_PATH = BASE_DIR / os.getenv("THRESHOLD_PATH", "artifacts/threshold.json")
    FEATURE_NAMES_PATH = BASE_DIR / os.getenv("FEATURE_NAMES_PATH", "artifacts/feature_names.json")

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()