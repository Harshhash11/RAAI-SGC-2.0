# Model and dataset cards

## RAAI-SGC technical corpus
- Scope: AI/ML, Cloud/DevOps, Cybersecurity, Software Engineering.
- Unit of analysis: 145 source articles; 4,771 graph associations and 290 draft QA records are linked annotations, **not** additional independent articles.
- Intended use: retrieval engineering, coursework, supervised source categorization after suitable review.
- Restrictions: source text and licensing must be reviewed; unverified draft answers must not be used as gold labels; sources can be incomplete and dated.
- Data provenance: Wikipedia page/revision references stored in `master.csv` and `data/processed/dataset_metadata.json`.
- Splits: 116 train, 14 validation, 15 test SOURCE records.

## Saved auxiliary domain-classifier artifact
- Path: `models/exports/domain_classifier.joblib`.
- Task: text → one of four IT subdomains. It is not conversational and does not generate answers.
- Architecture: tuned scikit-learn logistic regression pipeline. Exact hyperparameter search is in the executed ML notebook.
- Measured results: validation macro-F1 0.9365; test macro-F1 0.7667, accuracy 0.8000 on 15 examples.
- Caveats: small held-out test; topic overlap, Wikipedia article selection bias, and possible domain ambiguity. No confidence intervals or external domains are reported.

## Planned Qwen generative adapter (NOT PRESENT)
- Proposed base model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Intended task: evidence-grounded IT question answering with source citations.
- Training mechanism: 4-bit QLoRA with assistant-token-only supervised loss, only on genuine reviewed TRAIN examples.
- Status: Not trained in the distributed package because no CUDA GPU execution and no reviewed examples were available. No answer-quality score, adapter size, training loss or model card is manufactured.
- Once trained: save `adapter_config.json`, adapter weights, tokenizer config, experiment metadata, trainer logs, and human evaluation results to the same project folder.
