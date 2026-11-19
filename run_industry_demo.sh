#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv-llm/bin/python ]]; then
  echo 'ERROR: Missing .venv-llm environment. See docs/INDUSTRY_DEMO_FINAL.md' >&2
  exit 1
fi
if [[ ! -f data/processed/master.csv ]]; then
  echo 'ERROR: Missing data/processed/master.csv' >&2
  exit 1
fi
if [[ ! -f models/checkpoints/qlora_final_44/adapter_config.json ]]; then
  echo 'WARNING: Fine-tuned adapter not found. Extractive and pretrained Qwen modes can still work.' >&2
fi
.venv-llm/bin/python -m app.gradio_app
