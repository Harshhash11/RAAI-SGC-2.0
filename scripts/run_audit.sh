#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python dataset_engineering/validate_dataset.py
python -m pytest -q
python -m src.audit --csv data/processed/master.csv
