# RAAI-SGC 2.0: Local Qwen review-output format fix

Your first 10 attempted suggestions returned 9 `GENERATION_ERROR` and 1 `SUGGESTED`. The failures arose because the model returned alternate JSON field names (`question`/`answer`), key/value pairs, or invalid JSON quotation marks. This update modifies only the suggestion generator; no source data, human approvals, or model checkpoints are changed.

## Install inside the EXISTING main project (Ubuntu)

Stop the Gradio application first to free VRAM.

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
cp dataset_engineering/ai_assisted_review.py dataset_engineering/ai_assisted_review.py.before_format_fix
unzip -o ~/Downloads/RAAI_SGC_QA_OUTPUT_FORMAT_FIX.zip -d .
source .venv-llm/bin/activate
python -m unittest discover -s tests -p 'test_output_formats.py' -v
python dataset_engineering/ai_assisted_review.py --split train --limit 10 --resume --dry-run
python dataset_engineering/ai_assisted_review.py --split train --limit 10 --resume --max-new-tokens 256
```

The generation prompt now asks for three plain-text labeled lines:

```
QUESTION: <specific technical question>
ANSWER: <evidence-supported short answer, or INSUFFICIENT_EVIDENCE>
NOTES: <limitations / verification needed>
```

This prevents most JSON quoting errors; older JSON formats with `question`/`answer` or `suggested_question`/`suggested_answer` are also accepted. Arbitrary key/value pairs and incomplete outputs are not accepted as successful results. A template question or insufficient-evidence answer is marked `NEEDS_MANUAL_REVIEW` rather than `SUGGESTED`.

`--resume` preserves existing `SUGGESTED` records and reattempts only failed records among the selected first 10. It does not set `human_review_status=APPROVED` and does not modify `QA_REVIEW_QUEUE.csv` or exported training data. Do not approve machine-generated text without checking the source.

To count statuses after the run:

```bash
python - <<'PY'
import csv
from collections import Counter
with open('data/review/QA_AI_SUGGESTIONS.csv',encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
print(dict(Counter(r['generation_status'] for r in rows)))
for r in rows:
    if r['generation_status']=='GENERATION_ERROR':
        print(r['question_id'], r['ai_review_notes'][:240])
PY
```

The Python tests validate parsing and file-handling paths; the RTX 3060 generation must be executed on your laptop. There are no claimed GPU outputs in this update.
