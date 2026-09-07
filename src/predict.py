"""Inference par lot : ajoute une probabilite de churn a un CSV de clients.

    python src/predict.py --input data/nouveaux_clients.csv --output reports/predictions_batch.csv

Le CSV d'entree doit contenir les colonnes de features de configs/config.yaml ;
les autres colonnes (customerID, Churn...) sont ignorees pour la prediction et
recopiees telles quelles dans la sortie.
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.utils import coerce_features, load_config, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="CSV de clients a scorer")
    parser.add_argument("--output", required=True, help="CSV de sortie")
    parser.add_argument("--model", default=None, help="chemin du joblib (defaut : config)")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_path = resolve(args.model or cfg["artifacts"]["model_path"])
    if not model_path.exists():
        raise SystemExit(f"Modele introuvable : {model_path}. Lancer `make train` d'abord.")
    model = joblib.load(model_path)

    df = pd.read_csv(resolve(args.input))
    X = coerce_features(df, cfg["features"]["numeric"], cfg["features"]["categorical"])
    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["churn_pred"] = (proba >= args.threshold).astype(int)

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"{len(out)} lignes scorees -> {output}")
    print(f"Taux de churn predit : {out['churn_pred'].mean():.1%}")


if __name__ == "__main__":
    main()
