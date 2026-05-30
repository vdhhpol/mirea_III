"""
FastAPI сервис для кредитного скоринга.
Использует модель из src.model.
"""

import sys
from pathlib import Path
import time
import logging
import threading
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import predict
from src.config import config

# логирование
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# простые метрики
metrics = {
    "total_requests": 0,
    "success_requests": 0,
    "error_requests": 0,
    "total_time": 0.0  # суммарное время обработки в секундах
}
metrics_lock = threading.Lock()

# загрузка списка признаков
EXPECTED_FEATURES = None
if config.FEATURE_NAMES_PATH.exists():
    with open(config.FEATURE_NAMES_PATH, "r") as f:
        EXPECTED_FEATURES = json.load(f)
    logger.info("feature_names.json загружен, валидация полей включена")
else:
    logger.warning("feature_names.json не найден. Валидация полей будет ослаблена.")

# pydantic схемы
class Features(BaseModel):
    class Config:
        extra = "allow"   # разрешаем дополнительные поля

    @classmethod
    def validate_features(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if EXPECTED_FEATURES:
            missing = set(EXPECTED_FEATURES) - set(values.keys())
            if missing:
                raise ValueError(f"Отсутствуют обязательные поля: {missing}")
        return values

class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Вероятность дефолта (0..1)")
    prediction: int = Field(..., description="Бинарное предсказание (0/1) по порогу")
    risk_category: str = Field(..., description="Категория риска: low / medium / high")

# middleware для логирования всех запросов
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - status={response.status_code} - time={process_time:.3f}s")
        return response

# FastAPI приложение
app = FastAPI(title="Credit Scoring API", description="Оценка вероятности невозврата кредита", version="1.0.0")

app.add_middleware(LoggingMiddleware)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    with metrics_lock:
        avg_time = metrics["total_time"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0
        return {
            "total_requests": metrics["total_requests"],
            "success_requests": metrics["success_requests"],
            "error_requests": metrics["error_requests"],
            "average_response_time_sec": round(avg_time, 4)
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(features: Features, request: Request):
    start_time = time.time()
    with metrics_lock:
        metrics["total_requests"] += 1

    try:
        # валидация и предсказание
        features_dict = features.dict()
        Features.validate_features(features_dict)
        proba, binary_pred, risk = predict(features_dict)

        # обновляем метрики успеха
        with metrics_lock:
            metrics["success_requests"] += 1
            metrics["total_time"] += (time.time() - start_time)

        logger.info(f"Prediction OK: prob={proba:.4f}, pred={binary_pred}, risk={risk}, time={time.time()-start_time:.3f}s")
        return PredictionResponse(default_probability=proba, prediction=binary_pred, risk_category=risk)
    except Exception as e:
        with metrics_lock:
            metrics["error_requests"] += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)