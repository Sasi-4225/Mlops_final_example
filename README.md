# Heart Disease Prediction — End-to-End MLOps

Educational binary-classification project using the UCI Heart Disease dataset (Cleveland subset), scikit-learn, MLflow, Prefect, FastAPI, Docker, Prometheus, Grafana, GitHub Actions and Render.

> **Safety:** This is a learning demonstration, not a medical device. It must not be used to diagnose, rule out disease, triage patients, or guide treatment. Predictions are uncertain and require qualified clinical assessment.

## Architecture
1. Download and validate the UCI dataset (UCI dataset ID 45).
2. Preprocess numeric/categorical features and train a reproducible Random Forest pipeline.
3. Evaluate on a held-out test split; log parameters/metrics/model to MLflow when configured.
4. Save the serving pipeline as `artifacts/heart_disease_model.joblib`.
5. Serve predictions and health/Prometheus metrics via FastAPI.
6. Run locally or with Docker Compose; Grafana dashboards query Prometheus.
7. GitHub Actions runs tests; Render can build the Dockerfile.

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs. Use the example JSON shown by Swagger. The dataset is downloaded from UCI on first training; internet access is required.

## Docker
```powershell
docker build -t heart-disease-api:1.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0
```
Health: `http://localhost:8000/health`; docs: `http://localhost:8000/docs`; metrics: `http://localhost:8000/metrics`.

## Prometheus + Grafana
```powershell
docker compose -f docker-compose.monitoring.yml up --build -d
```
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (initial login admin / admin; change the password)
Stop: `docker compose -f docker-compose.monitoring.yml down`.

## MLflow (optional local tracking)
In terminal 1: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts`
In terminal 2: `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"; python train_model.py`
Open http://127.0.0.1:5000. Training still works without MLflow; tracking is skipped if the server is unavailable.

## Prefect
`python prefect_flow.py` runs the training task once. For local UI, start `prefect server start`, then in another terminal set `$env:PREFECT_API_URL="http://127.0.0.1:4200/api"` and run the flow.

## Render deployment
Create a Render **Web Service** from this repository, select Docker, and deploy. The Dockerfile trains the model during image build. Render sets PORT automatically. The service exposes `/health`, `/predict`, and `/docs`. Free instances may sleep when idle. Confirm the first deployment logs and test the live health endpoint.

## API contract
POST `/predict` accepts the 13 Cleveland input features:
```json
{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}
```
Returns `prediction` (0 = no disease label in this dataset; 1 = disease label present) and a model probability. This is a dataset label, not an individual's clinical diagnosis.

## Notes
- UCI target values 1–4 are grouped as positive; 0 is negative, following the common Cleveland binary target convention.
- The dataset is small and historically collected; performance does not establish clinical validity or fairness.
- Do not commit private patient data, credentials, MLflow databases, or generated model binaries.
