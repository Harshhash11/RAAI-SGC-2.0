#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -m pytest -q
printf '\nSetup complete. Start app: source .venv/bin/activate && python -m app.gradio_app\n'
