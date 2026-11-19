#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
python scripts/check_project.py
python dataset_engineering/validate_dataset.py >/dev/null
printf '\nChecks passed. GPU/QLoRA and manual review remain separate.\n'
