"""Tests des utilitaires : coercition des types, chargement, split, metriques, graphiques."""

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    ROOT,
    coerce_features,
    compute_metrics,
    load_config,
    load_data,
    save_evaluation_plots,
    split_data,
)

NUMERIC = ["tenure", "TotalCharges"]
CATEGORICAL = ["SeniorCitizen", "Contract"]


def test_coerce_features_handles_telco_quirks():
    df = pd.DataFrame(
        {
            "Contract": ["One year", "Two year", None],
            "TotalCharges": ["29.85", " ", "100"],  # l'espace vient du CSV Telco
            "SeniorCitizen": [0, 1, 0],
            "tenure": [1, 0, 12],
            "extra": ["ignore", "moi", "aussi"],
        }
    )
    X = coerce_features(df, NUMERIC, CATEGORICAL)

    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert X["TotalCharges"].dtype.kind == "f"
    assert np.isnan(X.loc[1, "TotalCharges"])
    assert X["SeniorCitizen"].tolist() == ["0", "1", "0"]
    assert X["Contract"].dtype == object
    assert pd.isna(X.loc[2, "Contract"])


@pytest.fixture
def small_cfg(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "customerID": [f"id{i}" for i in range(10)],
            "tenure": range(10),
            "TotalCharges": ["10", " ", "30", "40", "50", "60", "70", "80", "90", "100"],
            "SeniorCitizen": [0, 1] * 5,
            "Contract": ["One year", "Two year"] * 5,
            "Churn": ["Yes", "No"] * 5,
        }
    ).to_csv(csv, index=False)
    return {
        "data": {
            "csv_path": str(csv),
            "target": "Churn",
            "positive_label": "Yes",
            "drop_columns": ["customerID"],
            "test_size": 0.2,
            "random_state": 0,
        },
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    }


def test_load_data_and_split(small_cfg):
    X, y = load_data(small_cfg)
    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 5

    X_train, X_test, y_train, y_test = split_data(X, y, small_cfg)
    assert len(X_test) == 2
    assert len(X_train) + len(X_test) == len(X)
    # stratifie : une classe de chaque dans le test
    assert sorted(y_test.tolist()) == [0, 1]


def test_split_is_reproducible(small_cfg):
    X, y = load_data(small_cfg)
    _, test_a, _, _ = split_data(X, y, small_cfg)
    _, test_b, _, _ = split_data(X, y, small_cfg)
    assert test_a.index.tolist() == test_b.index.tolist()


def test_compute_metrics_perfect_and_keys():
    y = [0, 0, 1, 1]
    metrics = compute_metrics(y, [0.1, 0.2, 0.8, 0.9])
    assert set(metrics) == {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"}
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_save_evaluation_plots_writes_three_pngs(tmp_path):
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9, 0.7, 0.3])
    paths = save_evaluation_plots(y, proba, tmp_path)
    assert set(paths) == {"roc", "pr", "cm"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0


def test_project_config_is_consistent():
    """La config reelle du projet reste coherente avec le code."""
    cfg = load_config(ROOT / "configs" / "config.yaml")
    assert cfg["model"]["type"] in {"logreg", "random_forest"}
    assert not set(cfg["features"]["numeric"]) & set(cfg["features"]["categorical"])
    assert cfg["mlflow"]["tracking_uri"].startswith("sqlite:///")
