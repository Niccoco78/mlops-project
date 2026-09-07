"""API de prediction du churn (FastAPI).

Charge le pipeline produit par `src/train.py` (chemin MODEL_PATH, sinon celui de
la config) et expose :
  GET  /health         etat du service
  POST /predict        un client -> probabilite de churn
  POST /predict/batch  une liste de clients

Lancer : uvicorn service.app:app --reload   (ou `make serve`)
Doc interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.utils import ROOT, coerce_features, load_config

CONFIG = load_config(ROOT / "configs" / "config.yaml")
NUMERIC = CONFIG["features"]["numeric"]
CATEGORICAL = CONFIG["features"]["categorical"]
THRESHOLD = 0.5

app = FastAPI(
    title="Churn Classifier API",
    description="Probabilite qu'un client Telco resilie, d'apres le pipeline suivi par MLflow.",
    version="1.0.0",
)

_model = None


def reset_model_cache() -> None:
    """Oblige le prochain appel a recharger le modele (utile pour les tests)."""
    global _model
    _model = None


def get_model():
    """Chargement paresseux : au premier appel, pas au demarrage du serveur."""
    global _model
    if _model is None:
        path = Path(os.getenv("MODEL_PATH", ROOT / CONFIG["artifacts"]["model_path"]))
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modele introuvable : {path}. Lancer `make train` d'abord.",
            )
        _model = joblib.load(path)
    return _model


YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """Un client, avec les valeurs exactes du jeu Telco (visibles dans /docs)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
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
            ]
        }
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="anciennete en mois")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    # Inconnu pour un tout nouveau client : l'imputer du pipeline s'en charge.
    TotalCharges: float | None = Field(default=None, ge=0)


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    label: YesNo


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


def _predict(customers: list[Customer]) -> list[Prediction]:
    model = get_model()
    frame = pd.DataFrame([c.model_dump() for c in customers])
    proba = model.predict_proba(coerce_features(frame, NUMERIC, CATEGORICAL))[:, 1]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            churn=bool(p >= THRESHOLD),
            label="Yes" if p >= THRESHOLD else "No",
        )
        for p in proba
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(customers: list[Customer]) -> BatchPrediction:
    return BatchPrediction(predictions=_predict(customers))
