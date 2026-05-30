import pytest
import numpy as np
from src.model import predict, load_artifacts
from pathlib import Path
import pandas as pd
from src.model import predict

def test_artifacts_load():
    """Проверяем, что артефакты загружаются без ошибок"""
    preprocessor, model, calibrator, threshold = load_artifacts()
    assert preprocessor is not None
    assert model is not None
    assert calibrator is not None
    assert isinstance(threshold, float)

def test_predict_output_shape(sample_features):
    """Sanity-тест: predict возвращает кортеж (prob, pred, risk)"""
    prob, pred, risk = predict(sample_features)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert pred in (0, 1)
    assert risk in ("low", "medium", "high")

def test_predict_consistent():
    """Проверка детерминированности: повторный вызов даёт тот же результат"""
    import pandas as pd
    data_path = Path(__file__).parent.parent / "data" / "processed" / "X_test.csv"
    df = pd.read_csv(data_path)
    features = df.iloc[0].to_dict()
    prob1, pred1, risk1 = predict(features)
    prob2, pred2, risk2 = predict(features)
    assert prob1 == prob2
    assert pred1 == pred2
    assert risk1 == risk2