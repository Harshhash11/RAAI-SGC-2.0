# Completion plan for the main industry project

## What currently runs
1. Ubuntu TF-IDF + Markov–TF-IDF weighted graph retrieval and unweighted ablation.
2. Source-citing extractive Gradio chatbot (`python -m app.gradio_app`).
3. Automated integration/regression tests and dataset audit.
4. EDA/ML notebooks (auxiliary four-domain classifier, **not** LLM quality).
5. Three-mode retrieval evaluation code, **guarded** until independent gold questions are approved.
6. Colab T4 inference and QLoRA *implementation*, not a completed model experiment.

## Remaining evidence-producing work (in order)
1. Human review and licensing: source/QA review queues; record reviewer and decision.
2. Collect an **independent** test set; fill `data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv` and approve it.
3. Run `python -m evaluation.compare_retrievers` for the three modes on the same question set.
4. Run Colab T4 notebook 06 from the same Google Drive project folder; record model name, dependency versions, prompt, seed, generation settings and output.
5. Produce approved TRAIN-only `sft_train.jsonl` with `python -m training.prepare_sft` (never train on draft QAs).
6. QLoRA fine-tune only after review gate; save adapter into `models/checkpoints` and preserve training logs.
7. Run identical held-out prompts before/after fine-tuning; obtain human faithfulness/correctness judgments.
8. Add reviewed experimental tables/figures to Industry Immersion report and separate retrieval-evaluation research paper.

## Research protocol
Hold document corpus, prompt, top-k and retriever settings constant. Compare vector-only,
unweighted one-hop graph expansion, and Markov–TF-IDF weighted one-hop expansion.
Measure retrieval Recall@K/MRR and latency with approved independent gold. For answering,
measure factuality/faithfulness/citation accuracy with explicit human rubric. Avoid claiming
that source-document classification accuracy proves graph-augmented QA accuracy.

## Not yet completed
No confirmed successful model inference on Colab, no trained QLoRA adapter, no approved
training examples, no independently reviewed gold test questions, no GPU-reported results.
These milestones require user/runtime execution and human reviewers.
