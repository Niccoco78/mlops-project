"""Tests de l'API FastAPI, avec un petit pipeline entraine a la volee.

On n'utilise pas artifacts/model.joblib : les tests doivent passer sur un clone
vide, avant meme `make train`.
"""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.pipeline import build_pipeline
from src.utils import ROOT, coerce_features, load_config

CLIENT_EXEMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs" / "config.yaml")
    numeric, categorical = cfg["features"]["numeric"], cfg["features"]["categorical"]

    # Quelques clients synthetiques : les longs contrats restent, les autres partent.
    rows, y = [], []
    for i in range(12):
        row = dict(CLIENT_EXEMPLE, tenure=i * 6, MonthlyCharges=20.0 + 5 * i)
        row["Contract"] = "Two year" if i % 2 else "Month-to-month"
        row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
        rows.append(row)
        y.append(0 if i % 2 else 1)
    X = coerce_features(pd.DataFrame(rows), numeric, categorical)
    model_path = tmp_path / "model.joblib"
    joblib.dump(build_pipeline(numeric, categorical, "logreg").fit(X, y), model_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import service.app as service_app

    service_app.reset_model_cache()
    return TestClient(service_app.app)


def test_health(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_predict_returns_probability(client):
    reponse = client.post("/predict", json=CLIENT_EXEMPLE)
    assert reponse.status_code == 200
    corps = reponse.json()
    assert 0.0 <= corps["churn_probability"] <= 1.0
    assert corps["churn"] in (True, False)
    assert corps["label"] in ("Yes", "No")


def test_predict_accepts_new_customer_without_total_charges(client):
    nouveau = dict(CLIENT_EXEMPLE, tenure=0, TotalCharges=None)
    assert client.post("/predict", json=nouveau).status_code == 200


def test_predict_rejects_invalid_category(client):
    faux = dict(CLIENT_EXEMPLE, Contract="Weekly")
    assert client.post("/predict", json=faux).status_code == 422


def test_predict_batch(client):
    reponse = client.post("/predict/batch", json=[CLIENT_EXEMPLE, CLIENT_EXEMPLE])
    assert reponse.status_code == 200
    assert len(reponse.json()["predictions"]) == 2


def test_predict_without_model_is_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.joblib"))
    import service.app as service_app

    service_app.reset_model_cache()
    reponse = TestClient(service_app.app).post("/predict", json=CLIENT_EXEMPLE)
    assert reponse.status_code == 503
