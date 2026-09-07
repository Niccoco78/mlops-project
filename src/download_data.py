"""Telecharge le jeu Telco Customer Churn a l'emplacement indique par la config.

Le CSV n'est pas versionne (data/ est dans .gitignore) : ce script rend le
projet reproductible depuis un clone vide.
"""

from __future__ import annotations

import argparse
import urllib.request

import pandas as pd

from src.utils import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="retelecharge meme si present")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dest = resolve(cfg["data"]["csv_path"])
    if dest.exists() and not args.force:
        print(f"Deja present : {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Telechargement -> {dest}")
        urllib.request.urlretrieve(cfg["data"]["url"], dest)

    df = pd.read_csv(dest)
    print(f"{df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("Cible :", df[cfg["data"]["target"]].value_counts().to_dict())


if __name__ == "__main__":
    main()
