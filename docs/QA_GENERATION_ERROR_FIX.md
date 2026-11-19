# AI-assisted Q&A correction: diagnostic and retry patch

This is an **in-place update** to the same RAAI-SGC project; it does not modify the master CSV, original review queue, or approval fields.

## Installation (Ubuntu)

Stop any Qwen/Gradio process first. Extract the ZIP into `~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY` and activate `.venv-llm`.

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
cp dataset_engineering/ai_assisted_review.py dataset_engineering/ai_assisted_review.py.before_retry_fix
unzip -o ~/Downloads/RAAI_SGC_QA_GENERATION_RETRY_FIX.zip -d .
source .venv-llm/bin/activate
python -m unittest discover -s tests -p 'test_retry_errors.py' -v
```

## Inspect previous error details

```bash
python - <<'PY'
import csv
from collections import Counter
p='data/review/QA_AI_SUGGESTIONS.csv'
with open(p,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
print('Statuses:',dict(Counter(r.get('generation_status') for r in rows)))
for r in rows[:3]:
 print(r.get('question_id'), '=>', r.get('ai_review_notes'))
PY
```

## Retry the initial 10 failed rows without duplicates

```bash
python dataset_engineering/ai_assisted_review.py --split train --limit 10 --resume --max-new-tokens 512
```

If failures persist, the updated script prints the underlying exception directly. Share the first error and stop before increasing to 40 or all questions. Failed records remain `GENERATION_ERROR`; no human-review approvals are set.

## Details

- Existing `SUGGESTED` results are preserved during `--resume`.
- `GENERATION_ERROR` results are retried and replaced in-place by question ID, not duplicated.
- Human approval is never auto-assigned.
- The default response budget increases from 240 to 512 tokens, which may help with truncated JSON. It is **not** a guarantee against malformed responses or CUDA failures.
- The script adds a short excerpt of invalid model output to its diagnostic notes; inspect these notes before sharing them publicly.
- GPU generation has not been executed in this environment. The eight tests cover parsing, selection, output safety, and retry behavior.
