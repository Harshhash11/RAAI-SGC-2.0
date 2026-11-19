# RAAI-SGC 2.0 — Technical completion report (evidence-based)

## Research objective
Evaluate whether Markov–TF-IDF weighted concept-graph expansion improves evidence retrieval and grounded IT question answering compared with TF-IDF vector-only and unweighted graph expansion, holding source corpus and generation model fixed. This is **not** the previous Wikipedia graph construction paper.

## Architecture
145 Wikipedia-derived technical source articles → TF-IDF/sentence-embedding index → optional graph expansion (4,771 Markov–TF-IDF-weighted edges) → top-k sources → extractive answer or pretrained instruction LLM → source URLs and retrieval diagnostics. Gradio provides the Ubuntu demo; Google Colab T4 notebook supports the pretrained LLM and guarded QLoRA.

## Corpus and experimental isolation
5,206 master CSV records = 145 SOURCE + 4,771 EDGE + 290 QA_DRAFT. SOURCE split is 116 train, 14 validation, 15 test. Draft answers are **not approved** as supervised labels. Use source-level split when training; expert-reviewed independent evaluation questions must not be included in SFT examples. Retrieved documents can be indexed across splits for a retrieval task, but training targets must remain held out.

## Executed and reproducible engineering results
- Dataset structural checks: passed.
- 23 original regression tests: passed before this audit; additional provenance tests are included.
- 145 title-derived self-retrieval probes: vector-only 88.97%, unweighted graph 94.48%, weighted graph 94.48% top-1 self retrieval. These are **not** an external benchmark and do **not** show weighting superiority.
- Auxiliary four-domain source classifier: validation macro-F1 0.9365, held-out test accuracy 12/15 = 80.0%, test macro-F1 0.7667. This is not LLM question-answering quality.
- University-format EDA/ML notebooks contain saved local outputs; charts and interpretations need normal academic review.

## Missing scientific experiments
- Run Qwen/Qwen2.5-1.5B-Instruct on a Colab T4 and save the actual inference record; no pretrained generative GPU run has been performed in this packaging environment.
- Expert review source license/attribution and QA drafts; export only approved TRAIN records. `training/validation.py` rejects outputs not matching this approved export.
- Only after approval, launch QLoRA and save adapter, trainer loss logs, model revision and GPU evidence. Loss alone does not establish factual competence.
- Have reviewers create independent validation/test questions. Measure retrieval Recall@K/MRR, answer correctness and groundedness on the same held-out set, across the three modes with the same generator.

## Limitations
Small, Wikipedia-derived IT corpus; partially cleaned article text; 15-document auxiliary ML test set; no expert-approved QA or external evaluation yet; semantic backend requires a model download; Gradio is only verified in the user’s Ubuntu session, not in continuous deployment. Data reuse may require Wikipedia attribution/license handling.

## Claim discipline
The project **implements** weighted graph retrieval and a guarded fine-tuning pipeline. It does **not currently demonstrate** accuracy gains over unweighted graph retrieval, nor does it include a saved LLM fine-tuned adapter. Do not describe title self-matching as independent QA performance or the domain classifier as a fine-tuned LLM.
