#!/usr/bin/env bash
# RAAI-SGC 2.0: all commands run relative to the script, not current shell location.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f requirements.txt || ! -f data/processed/master.csv ]]; then
  echo "ERROR: Project files missing. Extract the COMPLETE ZIP into one directory." >&2
  exit 2
fi
usage(){
  cat <<'EOF'
Usage: ./start.sh setup|check|chat|audit|help
  setup  Create local .venv, install app dependencies and test
  check  Run unit tests and verify the dataset/retriever
  chat   Launch the Gradio chatbot on http://127.0.0.1:7860
  audit  Save honest project-readiness report
Run: ./start.sh setup && ./start.sh chat
Actual generative QLoRA training requires Colab T4 and reviewed training rows.
EOF
}
case "${1:-help}" in
  setup)
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements-app.txt
    .venv/bin/python -m pytest -q
    ;;
  check)
    [[ -x .venv/bin/python ]] || { echo 'First run: ./start.sh setup' >&2; exit 2; }
    .venv/bin/python -m pytest -q
    .venv/bin/python scripts/check_project.py
    ;;
  chat)
    [[ -x .venv/bin/python ]] || { echo 'First run: ./start.sh setup' >&2; exit 2; }
    exec .venv/bin/python -m app.gradio_app
    ;;
  audit)
    [[ -x .venv/bin/python ]] || { echo 'First run: ./start.sh setup' >&2; exit 2; }
    .venv/bin/python scripts/final_audit.py
    ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
