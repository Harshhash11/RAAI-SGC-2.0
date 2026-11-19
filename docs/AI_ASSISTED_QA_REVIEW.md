# AI-assisted draft Q&A correction (RTX 3060)

This utility **proposes** corrections grounded in the existing `evidence_quote` field. It does **not** validate citations against live Wikipedia, verify factual truth independently, label samples approved, or modify `QA_REVIEW_QUEUE.csv`. It will not train a model or produce a fake adapter.

## Install into the existing project

Extract the small update ZIP into `~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY/` (preserving paths). The script uses the same Hugging Face weights already downloaded on your Ubuntu machine. Activate `.venv-llm`; no new large package download should be necessary.

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
source .venv-llm/bin/activate
python dataset_engineering/ai_assisted_review.py --dry-run --split train --limit 10
python dataset_engineering/ai_assisted_review.py --split train --limit 10
```

Results appear in `data/review/QA_AI_SUGGESTIONS.csv`. Expect generation to consume GPU VRAM and run for multiple questions; check with `nvidia-smi`. Each processed row is saved immediately. It is safe to interrupt and resume.

```bash
python dataset_engineering/ai_assisted_review.py --split train --limit 40 --resume
# For the entire flagged training split, when satisfied with the first batch:
python dataset_engineering/ai_assisted_review.py --split train --limit 0 --resume
```

By default, only flagged questions in the **training split** are selected; validation/test examples are excluded. To propose edits for every training question, use `--all-questions`. This does not merge across splits.

Use `--dry-run` to inspect selection before consuming GPU, `--resume` to skip stored records, or `--overwrite` only if you're intentionally discarding prior AI suggestions. Inference is deterministic (`do_sample=False`), but outputs are not guaranteed identical across environments.

## Manual verification

Open `QA_AI_SUGGESTIONS.csv` alongside `data/review/QA_REVIEW_QUEUE.csv`. Verify the suggested wording, answer, and original evidence, and check the authoritative source, domain, and license. The model can hallucinate or produce malformed JSON; such records are marked `GENERATION_ERROR` or `NEEDS_MANUAL_REVIEW`.

After you personally verify a record, **manually copy approved wording** into the matching `question_id` row in `QA_REVIEW_QUEUE.csv`, use your own reviewer identifier and true review date, and set project-required review values only if warranted. Approval also requires source review in `SOURCE_REVIEW_QUEUE.csv`, including licensing and attribution checks. The official export workflow remains `python dataset_engineering/export_approved.py` followed by `python -m training.prepare_sft`. The proposal CSV is intentionally not used by those scripts.

For scientific validity, do not approve based only on model-generated proposals. Keep the train/validation/test splits isolated and avoid tuning to held-out tests.

## Why Qwen output may fail

The model is asked for a JSON object with three fields. A short `max_new_tokens` budget or a non-JSON answer can cause `GENERATION_ERROR`; the proposal still records the error and the next row continues. Keep batches modest on a 6 GB laptop GPU. If your laptop runs out of VRAM, close any concurrently running Qwen chatbot and retry.
