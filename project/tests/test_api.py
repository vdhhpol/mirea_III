import subprocess
import time
import requests
import pytest
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="module")
def live_server():
    """Запускает uvicorn в подпроцессе и останавливает после тестов"""
    app_path = Path(__file__).parent.parent / "src" / "app.py"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(5)
    yield
    proc.terminate()
    proc.wait()

def test_health_check(live_server):
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_valid_input(live_server, sample_features):
    resp = requests.post(f"{BASE_URL}/predict", json=sample_features)
    assert resp.status_code == 200
    data = resp.json()
    assert "default_probability" in data
    assert "prediction" in data
    assert "risk_category" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["risk_category"] in ("low", "medium", "high")

def test_predict_missing_field(live_server):
    incomplete = {"AMT_INCOME_TOTAL": 100000}
    resp = requests.post(f"{BASE_URL}/predict", json=incomplete)
    assert resp.status_code != 200