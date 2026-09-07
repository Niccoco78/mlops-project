"""Entrainement du classifieur de churn.

Deux modes :
  - baseline  (--no-tuning) : le pipeline avec ses hyperparametres par defaut ;
  - gridsearch (defaut)     : GridSearchCV sur la grille de configs/config.yaml.

Dans les deux cas, tout est suivi par MLflow (autolog + metriques de test), le
meilleur pipeline est sauve en joblib pour l'API et enregistre dans le Model
Registry sous l'alias "staging".
"""

from __future__ import annotations

import argparse
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_param_grid, build_pipeline
from src.utils import (
    compute_metrics,
    load_config,
    load_data,
    resolve,
    setup_mlflow,
    split_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--no-tuning", action="store_true", help="entraine la baseline, sans GridSearchCV"
    )
    parser.add_argument(
        "--no-register", action="store_true", help="n'enregistre pas le modele dans le registre"
    )
    return parser.parse_args()


def register_model(model_info, cfg: dict) -> None:
    """Pose l'alias "staging" sur la version qui vient d'etre enregistree.

    Les stages historiques (Staging / Production) sont deprecies depuis MLflow 2.9
    au profit des alias. On pose l'alias, un tag equivalent, et on tente la
    transition de stage demandee par le sujet sans en faire une condition d'echec.
    """
    name = cfg["mlflow"]["registered_model_name"]
    version = model_info.registered_model_version
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, "staging", version)
    client.set_model_version_tag(name, version, "stage", "Staging")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # depreciation connue, assumee
            client.transition_model_version_stage(name, version, stage="Staging")
    except Exception as exc:  # noqa: BLE001 - API en voie de suppression
        print(f"Transition de stage non disponible ({exc.__class__.__name__}), alias pose.")
    print(f"Registre : {name} version {version} -> alias 'staging'")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    uri, experiment = setup_mlflow(cfg)
    print(f"MLflow : {uri} | experience '{experiment}'")

    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)
    numeric, categorical = cfg["features"]["numeric"], cfg["features"]["categorical"]
    model_type = cfg["model"]["type"]
    pipe = build_pipeline(numeric, categorical, model_type)

    # autolog capture les hyperparametres, les metriques d'entrainement et, pour
    # GridSearchCV, un run enfant par combinaison testee. Le modele est logge a
    # la main plus bas, une seule fois, avec sa signature et son enregistrement.
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False, silent=True)

    run_name = f"{'baseline' if args.no_tuning else 'gridsearch'}-{model_type}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"stage": "training", "model_type": model_type})
        mlflow.log_params(
            {
                "tuning": not args.no_tuning,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_size": cfg["data"]["test_size"],
                "random_state": cfg["data"]["random_state"],
            }
        )

        if args.no_tuning:
            pipe.fit(X_train, y_train)
            best = pipe
        else:
            cv = StratifiedKFold(
                n_splits=cfg["cv"]["n_splits"],
                shuffle=True,
                random_state=cfg["data"]["random_state"],
            )
            grid = GridSearchCV(
                pipe,
                build_param_grid(cfg["model"]["params"]),
                cv=cv,
                scoring=cfg["cv"]["scoring"],
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            mlflow.log_metric(f"best_cv_{cfg['cv']['scoring']}", grid.best_score_)
            mlflow.log_dict(grid.best_params_, "best_params.json")
            print(f"Meilleurs parametres : {grid.best_params_}")
            print(f"Meilleur {cfg['cv']['scoring']} (CV) : {grid.best_score_:.4f}")

        proba = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        print("Jeu de test :", json.dumps({k: round(v, 4) for k, v in metrics.items()}))

        # Pipeline pour l'API et evaluate.py
        model_path = resolve(cfg["artifacts"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, model_path)
        print(f"Pipeline sauve : {model_path}")

        # Modele MLflow (avec signature) + Model Registry
        signature = infer_signature(X_train.head(), best.predict_proba(X_train.head()))
        model_info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            # MLflow 3 serialise en skops par defaut et refuse numpy.dtype comme type
            # "non fiable" ; cloudpickle est le format historique, sans cette audit.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=None
            if args.no_register
            else cfg["mlflow"]["registered_model_name"],
        )
        if not args.no_register:
            register_model(model_info, cfg)

        print(f"Run MLflow : {run.info.run_id}")


if __name__ == "__main__":
    main()
