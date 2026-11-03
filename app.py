from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_disease_api")
MODEL_PATH = Path("artifacts/heart_disease_model.joblib")
REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "HTTP request duration", ["method", "path"])
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0",
              description="Educational prediction demo only; not for medical decisions.")


class HeartInput(BaseModel):
    age: float = Field(gt=0, le=120)
    sex: int = Field(ge=0, le=1, description="Dataset coding: 1=male, 0=female")
    cp: int = Field(ge=1, le=4, description="Chest pain type (dataset code)")
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=7)


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact missing. Run python train_model.py first.")
    return joblib.load(MODEL_PATH)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        LATENCY.labels(request.method, request.url.path).observe(elapsed)
        logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                    request_id, request.method, request.url.path, status, elapsed * 1000)


@app.get("/")
def root():
    return {"service": "heart-disease-prediction", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health():
    if not MODEL_PATH.exists():
        return JSONResponse(status_code=503, content={"status": "error", "message": "model missing"})
    return {"status": "ok", "service": "heart-disease-prediction"}


@app.post("/predict")
def predict(payload: HeartInput):
    try:
        bundle = load_bundle()
        row = pd.DataFrame([payload.model_dump()])[bundle["features"]]
        model = bundle["model"]
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        return {
            "prediction": prediction,
            "label": "dataset_positive" if prediction == 1 else "dataset_negative",
            "probability_positive": round(probability, 4),
            "notice": "Educational model output only; not a diagnosis or medical advice."
        }
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
