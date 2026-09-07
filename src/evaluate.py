"""Evaluation finale sur le jeu de test.

Recharge le pipeline entraine (joblib par defaut, ou depuis le Model Registry avec
--from-registry), calcule les metriques, trace ROC / precision-rappel / matrice de
confusion, ecrit les predictions pour l'analyse d'erreurs, et logge le tout comme
artefacts d'un run MLflow "evaluation".
"""

from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report

from src.utils import (
    compute_metrics,
    load_config,
    load_data,
    resolve,
    save_evaluation_plots,
    setup_mlflow,
    split_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="chemin du joblib (defaut : config)")
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="charge models:/<registered_model_name>@staging au lieu du joblib",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_model(args: argparse.Namespace, cfg: dict):
    if args.from_registry:
        uri = f"models:/{cfg['mlflow']['registered_model_name']}@staging"
        print(f"Modele : {uri}")
        return mlflow.sklearn.load_model(uri), uri
    path = resolve(args.model or cfg["artifacts"]["model_path"])
    if not path.exists():
        raise SystemExit(f"Modele introuvable : {path}. Lancer `make train` d'abord.")
    print(f"Modele : {path}")
    return joblib.load(path), str(path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_mlflow(cfg)

    X, y = load_data(cfg)
    _, X_test, _, y_test = split_data(X, y, cfg)
    model, source = load_model(args, cfg)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= args.threshold).astype(int)
    metrics = compute_metrics(y_test, proba, args.threshold)

    reports_dir = resolve(cfg["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_plots(y_test, proba, reports_dir, args.threshold)

    # Predictions ligne a ligne, pour regarder ou le modele se trompe
    predictions = X_test.copy()
    predictions["y_true"] = y_test.to_numpy()
    predictions["churn_probability"] = proba.round(4)
    predictions["y_pred"] = pred
    predictions["error"] = predictions["y_true"] != predictions["y_pred"]
    predictions.to_csv(reports_dir / "predictions.csv", index=False)

    report = classification_report(y_test, pred, target_names=["No", "Yes"], digits=4)
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tags({"stage": "evaluation", "model_source": source})
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifacts(str(reports_dir), artifact_path="evaluation")

    print()
    print(pd.Series(metrics).round(4).to_string())
    print()
    print(report)
    print(f"Rapports ecrits dans {reports_dir}")


if __name__ == "__main__":
    main()
