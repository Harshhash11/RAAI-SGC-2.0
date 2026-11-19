# How to maintain the one project folder

## KEEP (do not delete)
- `data/processed/master.csv` — canonical dataset and 32-column schema.
- `data/review/` — source/term/Q&A review queues and gold evaluation template.
- `src/hybrid.py` — vector and graph retrieval implementations.
- `src/answers.py`, `src/llm.py`, `app/gradio_app.py` — chatbot logic.
- `training/`, `evaluation/`, `tests/`, `docs/`, `notebooks/` — scientific reproducibility.
- `dataset_engineering/` — Part 1 validation, export and collector reference.

## SAFE TO DELETE/RECREATE
- `.venv/` — reinstall using `scripts/setup_ubuntu.sh`.
- `__pycache__/`, `.pytest_cache/` — automatically recreated.
- `models/checkpoints/` — only if not needed; contains future training runs.
- `reports/generated/` — generated experiments; archive any results used in the paper before deletion.

## ADD FILES HERE
- More validated sources: prepare a separately versioned dataset and replace `data/processed/master.csv` only after passing validation. Do not append arbitrary EDGE/QA rows without rebuilding all relationships.
- Human-reviewed approvals: update review queues in `data/review`, then run a provenance-preserving exporter.
- Independent benchmark: `data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv`.
- Additional evaluation scripts: `evaluation/`.
- Model checkpoint/adapters: `models/checkpoints/`.
- Research paper drafts written in your own words: `docs/research/`.

## One-time migration from your older folders
You can **archive**, not immediately delete, `RAAI_SGC_PART2_STARTER`, `RAAI_SGC_PART2_HYBRID_UBUNTU`, and `RAAI_SGC_PART1_COMPLETE` after verifying this unified package works. This ZIP contains the canonical Part 1 master and review files, the hybrid retrieval code, and university notebooks. The old V2/V3.x test folders are not required for normal operation. Preserve at least one offline backup of all old folders and any manually reviewed files before deletion.

## Safe steps on Ubuntu
```bash
cp -a RAAI_SGC_2_COMPLETE_PROJECT RAAI_SGC_2_COMPLETE_PROJECT_BACKUP
cd RAAI_SGC_2_COMPLETE_PROJECT
source .venv/bin/activate
python -m pytest -q
```
Never run bulk `rm -rf` commands copied from chat without first reviewing the target path.
