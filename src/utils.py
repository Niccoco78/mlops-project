"""Fonctions partagees : configuration, donnees, split, MLflow, metriques, graphiques."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenetre : on ecrit des fichiers

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Racine du projet : les chemins de la config sont relatifs a elle, quel que
# soit le dossier depuis lequel on lance les scripts.
ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- configuration
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Chemin absolu : tel quel s'il l'est deja, sinon relatif a la racine du projet."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ----------------------------------------------------------------------- donnees
def coerce_features(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Met les colonnes dans le type attendu par le pipeline, dans l'ordre attendu.

    - numeriques : to_numeric avec errors="coerce", puis float. Dans Telco,
      TotalCharges est lu comme du texte parce que 11 lignes contiennent un espace
      (clients a tenure 0) : ces valeurs deviennent NaN et l'imputer du pipeline
      s'en charge. Le float evite aussi qu'une colonne entiere (tenure) ne puisse
      pas porter de NaN en inference, ce que la signature MLflow verifie.
    - categorielles : chaines Python (object) et NaN pour les manquants. Cela evite
      les surprises entre versions de pandas (astype(str) transforme parfois NaN
      en "nan") et rend SeniorCitizen (0/1) homogene avec les autres colonnes.
    """
    df = df.copy()
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in categorical:
        df[col] = df[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return df[numeric + categorical]


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Lit le CSV brut et renvoie (X, y), y en 0/1."""
    data_cfg = cfg["data"]
    df = pd.read_csv(resolve(data_cfg["csv_path"]))
    df = df.drop(columns=data_cfg.get("drop_columns", []), errors="ignore")

    y = (df[data_cfg["target"]] == data_cfg["positive_label"]).astype(int)
    X = coerce_features(df, cfg["features"]["numeric"], cfg["features"]["categorical"])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """Split stratifie et reproductible. Le meme random_state est utilise par train.py
    et evaluate.py, donc les deux scripts voient exactement le meme jeu de test."""
    data_cfg = cfg["data"]
    return train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )


# ------------------------------------------------------------------------ MLflow
def setup_mlflow(cfg: dict) -> tuple[str, str]:
    """Configure le tracking MLflow. Priorite : variables d'environnement (.env) > config."""
    load_dotenv(ROOT / ".env")
    ml_cfg = cfg["mlflow"]
    uri = os.getenv("MLFLOW_TRACKING_URI", ml_cfg["tracking_uri"])
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", ml_cfg["experiment_name"])

    # Un chemin SQLite relatif serait interprete depuis le dossier courant :
    # on l'ancre a la racine du projet.
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix) :]).is_absolute():
        uri = prefix + (ROOT / uri[len(prefix) :]).as_posix()

    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(experiment) is None:
        client.create_experiment(experiment, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return uri, experiment


# --------------------------------------------------------------------- metriques
def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


# ------------------------------------------------------------------- graphiques
def save_evaluation_plots(y_true, proba, out_dir: str | Path, threshold: float = 0.5) -> dict:
    """Courbe ROC, courbe precision-rappel et matrice de confusion en PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    paths = {}

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax, name="modele")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="hasard")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    paths["roc"] = out_dir / "roc_curve.png"
    fig.savefig(paths["roc"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax, name="modele")
    ax.axhline(y_true.mean(), linestyle="--", color="grey", label="hasard")
    ax.set_title("Courbe precision-rappel")
    ax.legend(loc="upper right")
    paths["pr"] = out_dir / "pr_curve.png"
    fig.savefig(paths["pr"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["No", "Yes"], colorbar=False, ax=ax
    )
    ax.set_title(f"Matrice de confusion (seuil {threshold})")
    paths["cm"] = out_dir / "confusion_matrix.png"
    fig.savefig(paths["cm"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    return paths
