from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "heart-disease-prediction"

def test_health():
    response = client.get("/health")
    assert response.status_code in (200, 503)

def test_predict_validation():
    response = client.post("/predict", json={"age": 10})
    assert response.status_code == 422

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "heart_api_requests_total" in response.text
