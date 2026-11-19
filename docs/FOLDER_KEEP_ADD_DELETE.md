# Project folder maintenance

**Canonical working directory:** `RAAI_SGC_2_COMPLETE_PROJECT` on Ubuntu and Google Drive.
This ZIP unpacks to that exact folder name. It is a **replacement snapshot**; back up
local changes before overwriting.

## Never delete
- `data/processed/master.csv` or `data/review/` (review decisions are irreplaceable)
- `src/`, `app/`, `evaluation/`, `training/`, `tests/`
- `notebooks/05_colab_t4_llm_qlora.ipynb`, `notebooks/06_colab_drive_inference_and_training.ipynb`

## Add here
- Approved train examples: `data/training/sft_train.jsonl` via gated exporter
- Independent gold tests: `data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv`
- Trained LoRA adapters: `models/checkpoints/<run_name>/`
- Evaluation JSON, generated plots: `reports/generated/`
- Paper and university report drafts: `docs/research/`
- Demo screenshots: `reports/demo_screenshots/`

## Safe to regenerate / delete after backup
- `.venv/`, `.pytest_cache/`, `__pycache__/`, `.ipynb_checkpoints/`
- Hugging Face cache only if disk space needed (will require re-download)

## Separate assignment folder
`RAAI_SGC_EDA_ML_COLAB_COMPLETED` is an **independent submission artifact**.
Do not copy its outputs into the main project unless deliberately incorporating
verified findings; it is not the main chatbot source tree.
