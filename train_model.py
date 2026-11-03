from pathlib import Path
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "heart_disease_model.joblib"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data():
    """Fetch UCI Heart Disease dataset (ID 45); no patient file is committed."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()
    X.columns = [str(c).strip().lower().replace(" ", "_") for c in X.columns]
    aliases = {"trestbps": "trestbps", "thalach": "thalach"}
    X = X.rename(columns=aliases)
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"UCI feature schema changed; missing: {missing}; got {list(X.columns)}")
    X = X[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    y = (y.astype(float) > 0).astype(int)
    if y.nunique() != 2:
        raise ValueError("Expected a binary target after mapping original target 0 vs 1-4.")
    return X, y


def build_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", prep),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ])


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logging.info("Evaluation: %s", metrics)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Heart_Disease_Prediction")
            with mlflow.start_run():
                mlflow.log_param("model", "RandomForestClassifier")
                mlflow.log_param("random_state", 42)
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as exc:
            logging.warning("MLflow logging skipped: %s", exc)
    return metrics


if __name__ == "__main__":
    train()
