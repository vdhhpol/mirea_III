import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from .config import config

_preprocessor = None
_model = None
_calibrator = None
_threshold = None

def load_artifacts():
    global _preprocessor, _model, _calibrator, _threshold
    if _preprocessor is None:
        _preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        _model = joblib.load(config.MODEL_PATH)
        _calibrator = joblib.load(config.CALIBRATOR_PATH)
        with open(config.THRESHOLD_PATH, 'r') as f:
            _threshold = json.load(f)['threshold']
        print("Артефакты загружены")
        print(f"Порог: {_threshold:.4f}")
    return _preprocessor, _model, _calibrator, _threshold

def predict(features: dict):
    preprocessor, model, calibrator, threshold = load_artifacts()
    df = pd.DataFrame([features])
    X_processed = preprocessor.transform(df)
    raw_proba = model.predict_proba(X_processed)[:, 1]
    proba_cal = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1][0]
    binary_pred = int(proba_cal >= threshold)
    if proba_cal < 0.2:
        risk = "low"
    elif proba_cal < 0.5:
        risk = "medium"
    else:
        risk = "high"
    return proba_cal, binary_pred, risk