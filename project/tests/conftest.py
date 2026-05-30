import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.model import predict, load_artifacts

@pytest.fixture(scope="session")
def sample_features():
    """Берём первую строку из X_test.csv"""
    data_path = Path(__file__).parent.parent / "data" / "processed" / "X_test.csv"
    df = pd.read_csv(data_path)
    features = df.iloc[0].to_dict()
    return features

@pytest.fixture(scope="session")
def preloaded_artifacts():
    """Загружаем артефакты один раз на сессию"""
    return load_artifacts()