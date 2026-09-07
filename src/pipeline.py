"""Construction du pipeline : pretraitement (ColumnTransformer) + modele."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Chaque entree est une fabrique : on veut un estimateur neuf a chaque appel.
MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
}


def build_pipeline(
    numeric: list[str], categorical: list[str], model_type: str = "logreg"
) -> Pipeline:
    """Pipeline complet : imputation + scaling / imputation + one-hot, puis le modele.

    Tout le pretraitement est dans le pipeline, donc il est appris sur le jeu
    d'entrainement uniquement (pas de fuite) et rejoue a l'identique en inference.
    """
    if model_type not in MODELS:
        raise ValueError(f"model_type inconnu : {model_type!r} (attendu : {sorted(MODELS)})")

    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" : une categorie jamais vue a l'entrainement
            # donne une ligne de zeros au lieu d'une erreur en production.
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(transformers=[("num", num, numeric), ("cat", cat, categorical)])

    return Pipeline(steps=[("pre", pre), ("model", MODELS[model_type]())])


def build_param_grid(params: dict | None) -> dict:
    """Prefixe les hyperparametres pour GridSearchCV sur le pipeline.

    {"C": [0.1, 1]} devient {"model__C": [0.1, 1]} ; une valeur seule est
    mise dans une liste pour que la grille reste valide.
    """
    grid = {}
    for key, values in (params or {}).items():
        grid[f"model__{key}"] = values if isinstance(values, list) else [values]
    return grid
