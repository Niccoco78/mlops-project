"""Tests du pipeline : structure, robustesse aux valeurs manquantes et inconnues, grille."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import MODELS, build_param_grid, build_pipeline


def _toy_data():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": ["x", "y", "x", np.nan, "y", "x", "y", "x"],
        },
        dtype=object,
    )
    X["a"] = X["a"].astype(float)
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    return X, y


def test_build_pipeline():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert "pre" in dict(pipe.named_steps)
    assert "model" in dict(pipe.named_steps)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="model_type inconnu"):
        build_pipeline(["a"], ["b"], "svm")


@pytest.mark.parametrize("model_type", sorted(MODELS))
def test_pipeline_handles_missing_and_unknown_values(model_type):
    X, y = _toy_data()
    pipe = build_pipeline(["a"], ["b"], model_type).fit(X, y)

    # NaN numerique, NaN categoriel et categorie jamais vue : aucune erreur attendue
    nouveau = pd.DataFrame({"a": [np.nan, 3.0], "b": ["z", np.nan]}, dtype=object)
    nouveau["a"] = nouveau["a"].astype(float)
    proba = pipe.predict_proba(nouveau)

    assert proba.shape == (2, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_build_param_grid_prefixes_and_wraps():
    grid = build_param_grid({"C": [0.1, 1.0], "penalty": "l2"})
    assert grid == {"model__C": [0.1, 1.0], "model__penalty": ["l2"]}
    assert build_param_grid(None) == {}


def test_param_grid_keys_are_pipeline_params():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    grid = build_param_grid({"C": [1.0], "solver": ["lbfgs"]})
    assert set(grid) <= set(pipe.get_params())


def test_each_call_returns_a_fresh_estimator():
    p1 = build_pipeline(["a"], ["b"])
    p2 = build_pipeline(["a"], ["b"])
    assert p1.named_steps["model"] is not p2.named_steps["model"]
