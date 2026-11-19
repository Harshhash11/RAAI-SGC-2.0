# Review and release control

1. Start with `output/SOURCE_REVIEW_QUEUE.csv`. Check each title against its revision-tracked source URL; inspect short-extract flags. Verify topical relevance, evidence quality, extract completeness, attribution and license applicability. Enter **APPROVED** or **REJECTED** independently in the relevant review fields, and provide a reviewer identifier and notes. A low-risk automated record is **not** approved automatically.
2. Review `output/TERM_REVIEW_QUEUE.csv` for incorrect concept normalization, especially generic tokens and confusing stemming/lemmatization. Use `canonical_override` only after examining evidence. It is intentionally separate from the frozen master CSV until reviewed changes can be rebuilt with correct graph weights.
3. Review `output/QA_REVIEW_QUEUE.csv`. Confirm factuality, answer relevance, completeness, and whether the quote supports the approved answer. Put a revised expert-approved response in `approved_answer`, plus reviewer metadata. The current questions are extractive drafts; source quotation is *not* factual validation.
4. Build the independent evaluation set from held-out questions written and checked independently of the training drafts. Fill `INDEPENDENT_EVALUATION_TEMPLATE.csv`. The template is empty by design.
5. Run `python3 src/export_approved.py`. It exports only `QA_DRAFT` rows with explicit source and answer approvals; initially it yields an empty file. **Do not claim final training readiness while that file is empty**.
6. Every subsequent review correction should receive a new SHA-256 and version; keep the input and output for reproducibility.

## Human audit acceptance criteria

- Every included training source has verified domain relevance, text completeness, factual accuracy, license/attribution review.
- Every included question and answer has been individually reviewed for factual accuracy and sufficient evidence; answer data is separated from evaluation questions by source group.
- Canonical graph terms that appear in retrieval are screened; weak terms may be excluded or downweighted only with documented rules and regeneration of normalized probabilities.
- The evaluation benchmark is genuinely independent, contains all four domains and representative question types, and has recorded annotators and disagreement resolution.

## Important use limitation

This package is a **complete reproducible Part 1 engineering deliverable**. Human content validation, copyright/attribution approval, and independent benchmark creation remain separate academic responsibilities. These cannot be safely simulated by automated code.
