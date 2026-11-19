# RAAI-SGC 2.0: Evaluating Markov–TF-IDF Weighted Graph-Augmented Retrieval for Domain-Specific LLM Question Answering

**Status:** Technical experimental report draft. Not a validated final research paper. The evidence below is genuine saved CPU output, but it does not establish answer-quality improvement or LLM performance. Author information, references, external benchmark adjudication and university formatting require human completion.

## Abstract (restricted to observations actually measured)

We implement a domain-focused retrieval-augmented question-answering prototype over four information-technology categories: artificial intelligence and machine learning, cybersecurity, cloud computing and DevOps, and software engineering. The dataset includes 145 source documents, 4,771 Markov–TF-IDF weighted graph edges, and 290 unapproved question-answer drafts. Three retrieval variants share the same TF-IDF baseline: vector-only, graph expansion with unweighted source-term links, and graph expansion with Markov–TF-IDF weights. A deterministic engineering diagnostic formed one question from each source title and measured self-retrieval: Recall@1 was 88.97% for vector-only and 94.48% for both graph variants; Recall@5 was 100% for all variants. These *title-based probes are not independent information-needs or a trustworthy estimate of question-answering performance*. The software supports evidence citation, an interactive Gradio prototype and optional pretrained model inference and QLoRA fine-tuning. The present environment did not run a pretrained generative LLM or QLoRA, and independent gold evaluation labels have not yet been approved. Consequently, no claims about improved answer correctness, faithfulness or generalization are warranted.

## 1. Problem and scope

Industry-specific technical assistants must retrieve domain-appropriate material before generating natural language answers. This study separates retrieval effectiveness from generation quality. A previous RAAI-SGC study constructed recursive semantic graphs; the new research question is whether its weighted edges improve retrieval in downstream domain-specific question answering under controlled retrieval and generation budgets. This is a distinct evaluation target rather than a repeat of graph creation.

## 2. Data and controls

The original mixed-record dataset has 5,206 rows and 32 columns, comprising SOURCE (145), EDGE (4,771) and QA_DRAFT (290) records. SOURCE document counts are AI/ML 40, Cloud/DevOps 32, Cybersecurity 36, Software Engineering 37. Source-level partitions comprise train 116, validation 14, and test 15. No human-approved source or QA evidence is currently recorded. The 290 draft QA records are not used as gold labels or training examples. Structural checks found no critical integrity or weight-formula errors; human provenance and licensing reviews are pending.

## 3. Retrieval methods

The base representation is TF-IDF using uni- and bi-grams and cosine similarity over article title and source text. Graph expansion obtains candidate source-title matches through terms associated with high-similarity anchor sources. The unweighted ablation assigns equal path strength; the weighted hybrid uses each association's original edge weight, derived as W(A,B) = P(A→B) × [1 + TFIDF(B)]. Ranking gives priority to document similarity and a bounded graph contribution. The same source corpus and retrieval cutoff are used for each comparison. The embedding backend is implemented but was not run here, so it is not included in the measured comparison.

## 4. Results from executed CPU experiments

| Method | Title self-retrieval Recall@1 | Title self-retrieval Recall@5 | Mean latency per query | Fraction of probes with graph activation |
|---|---:|---:|---:|---:|
| TF-IDF vector-only | 0.889655 | 1.000000 | 0.7882 ms | 0.000000 |
| Unweighted graph expansion | 0.944828 | 1.000000 | 0.7816 ms | 0.696552 |
| Markov–TF-IDF weighted graph expansion | 0.944828 | 1.000000 | 0.7777 ms | 0.696552 |

There are 145 mechanically generated, title-based questions in this diagnostic. The results verify code behavior but substantially favor title matching and cannot estimate real question-answering performance. The weighted and unweighted graph configurations tied on these self-retrieval measures; this experiment does not show a measured benefit specifically attributable to Markov–TF-IDF weighting. Latency values were measured on this execution environment, are unbatched and are not portable across hardware.

### Supporting auxiliary ML experiment

A separate four-domain source classifier, trained on the 116-source training split and tuned with validation data, selected tuned logistic regression with validation macro-F1 0.936508. Held-out test accuracy was 0.80 and macro-F1 was 0.766667 on only 15 documents. These numbers describe *domain classification*, not graph-augmented retrieval or generative question answering. The small sample size calls for cautious interpretation.

## 5. LLM and training status

The demonstration application generates extractive, evidence-backed text offline. Code for pretrained `Qwen/Qwen2.5-1.5B-Instruct` generation and a 4-bit QLoRA adapter is provided for a Colab T4 GPU. It has not been executed within this CPU-only packaging environment. No QLoRA training result exists; the training-label review gate currently contains zero approved QA records, so any putative trained adapter or loss curve would be fabricated. Gold evaluation, prompt/response samples, latency and safety reviews must be generated during real Colab runs and documented separately.

## 6. Validity limitations

1. Source-title self-retrieval is a software sanity check, not independent gold evaluation, and should not be published as retrieval accuracy without qualification.
2. The source dataset has only 145 documents across four technical subdomains.
3. Source licenses and content/QA correctness have not received expert approval.
4. There is no controlled generated-answer assessment (correctness, citation attribution, faithfulness, abstention and hallucination rate).
5. The weighted and unweighted graph methods have identical self-retrieval Recall@1/5 in the executed diagnostic.
6. Neither GPU inference nor supervised adaptation has been performed here.

## 7. Required confirmatory experiment

Obtain a disjoint, independently labeled test question set with answerable and unanswerable examples and annotated relevant sources. Compare vector-only, unweighted graph, weighted graph and a semantic embedding baseline at fixed top-k and identical generation model, prompts, sampling controls and evidence budget. Report Recall@k, MRR, correctness and groundedness with annotator agreement and per-domain breakdowns. Where feasible, use bootstrap confidence intervals on held-out queries. Run QLoRA only after genuine supervised data approval and compare a pretrained generator against the fine-tuned adapter on a held-out answer-quality set. Save all Colab outputs, package versions, identifiers, annotation decisions and model checkpoints.

## 8. Reproducibility

Recorded measurements: `reports/generated/retrieval_title_self_probes.json`, `reports/generated/measured_results.json`, `reports/generated/ml_domain_classification_metrics.json`, `reports/generated/executed_checks.json`. Executed university notebooks: `notebooks/01_RAAI_SGC_EDA_COLAB.ipynb` and `notebooks/02_RAAI_SGC_ML_COLAB.ipynb`. Code: `src/hybrid.py`, `evaluation/compare_retrievers.py`, `training/train_qlora_colab.py`. For practical instructions see `docs/REPRODUCIBILITY.md` and `docs/SUBMISSION_READINESS.md`.

**References must be supplied and verified by the actual authors. Do not submit this draft as a final peer-reviewed publication or as evidence that the GPU model has been trained.**
