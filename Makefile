# Fonctionne sous Linux, macOS et Windows (Git Bash) : le venv est appele par
# son chemin, sans `activate`. Sous Windows il faut GNU make (winget install ezwinports.make).

PY_SYS ?= python
ifeq ($(OS),Windows_NT)
PY = .venv/Scripts/python.exe
else
PY = .venv/bin/python
endif

CONFIG ?= configs/config.yaml
EXP ?= churn-exp
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
export MLFLOW_TRACKING_URI
export MLFLOW_EXPERIMENT_NAME = $(EXP)

.PHONY: help init data baseline train evaluate predict test lint format ui serve docker-build docker-run all clean

help:            ## liste les cibles
	@$(PY_SYS) -c "import re; [print(f'{n:14s} {d}') for n, d in sorted(re.findall(r'^([a-z-]+):[^#]*## *(.*)$$', open('Makefile').read(), re.M))]"

init:            ## cree le venv et installe les dependances
	$(PY_SYS) -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

data:            ## telecharge data/raw.csv
	$(PY) -m src.download_data --config $(CONFIG)

baseline:        ## modele avec hyperparametres par defaut, suivi MLflow
	$(PY) -m src.train --config $(CONFIG) --no-tuning

train:           ## GridSearchCV + MLflow + Model Registry
	$(PY) -m src.train --config $(CONFIG)

evaluate:        ## metriques, courbes et matrice de confusion dans reports/ et MLflow
	$(PY) -m src.evaluate --config $(CONFIG)

predict:         ## inference par lot sur data/raw.csv
	$(PY) -m src.predict --config $(CONFIG) --input data/raw.csv --output reports/predictions_batch.csv

test:            ## tests unitaires
	$(PY) -m pytest

lint:            ## verification ruff (style + imports)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:          ## reformate le code
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

ui:              ## interface MLflow sur http://localhost:5000
	$(PY) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

serve:           ## API FastAPI sur http://localhost:8000/docs
	$(PY) -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:    ## image Docker de l'API
	docker build -t churn-api:latest -f service/Dockerfile .

docker-run:      ## lance l'image sur le port 8000
	docker run --rm -p 8000:8000 churn-api:latest

all: data train evaluate test   ## enchaine tout depuis un clone vide (apres init)

clean:           ## supprime les fichiers regeneres (pas mlflow.db ni le modele)
	$(PY_SYS) -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in ['reports', '.pytest_cache', '.ruff_cache'] + glob.glob('**/__pycache__', recursive=True)]"
