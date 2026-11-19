# Submission readiness — factual audit

## Executed and included

- 145 source documents, 4,771 graph edges, 290 **unapproved draft** Q&A records in `data/processed/master.csv`.
- EDA and auxiliary domain-classification ML notebook outputs are saved in `notebooks/01_RAAI_SGC_EDA_COLAB.ipynb` and `notebooks/02_RAAI_SGC_ML_COLAB.ipynb`.
- The auxiliary domain classification model is saved at `models/exports/domain_classifier.joblib`; this is **not** a fine-tuned language model.
- Regression tests, project checks, and retriever title self-probes have run. Execution logs and measured metrics are in `reports/generated`.
- The TF-IDF vector, unweighted, and Markov–TF-IDF weighted graph retrievers all ran on the corpus.
- Gradio interface source is included; original live demonstration was verified on Ubuntu, but this packaging environment did **not** perform browser-level end-to-end testing.

## Not executed; DO NOT claim completed

- Sentence-transformer semantic embedding model download / inference (not available in this runtime).
- Pretrained generative Qwen LLM inference on Colab T4 (there is no CUDA GPU in this packaging runtime).
- 4-bit QLoRA fine-tuning (no CUDA GPU **and** zero human-approved supervised Q&A rows).
- Research-quality Recall@K/MRR comparison (no independently approved gold benchmark questions).
- Answer correctness, faithfulness, hallucination or domain-generalization performance; not measured.
- Human source quality, citation licensing clearance, and expert approval of annotations.

## Required prior to academic submission

1. Have legitimate domain reviewers approve sources and supervise Q&A, documenting who reviewed what and when. Do not mark review columns approved without real review.
2. Build an independent, held-out, annotated benchmark for retrieval and generated-answer evaluation; report actual metrics including uncertainty.
3. Run `notebooks/06_colab_drive_inference_and_training.ipynb` on a real T4 to establish pretrained model inference. Fine-tune only after approved training data exists.
4. Compare LLM-only/RAG/vector and graph-augmented approaches with controlled model, corpus and prompts, and record real GPU execution logs.
5. Recheck university template, ethics, source licensing, authorship declaration, personal details, references and presentation demonstration.

**This folder is a complete executable project scaffold with real CPU experiments, but not a finished scientifically validated or GPU-trained final submission.**
