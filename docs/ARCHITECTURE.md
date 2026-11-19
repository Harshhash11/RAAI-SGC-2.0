# Architecture

```
Part 1 master.csv
  ├── SOURCE (145) ──> TF-IDF / optional semantic embedding ──┐
  ├── EDGE (4771) ──> weighted graph expansion + ablation ────┤
  └── QA_DRAFT (290) ──> REVIEW ONLY, excluded from training    │
                                                               ▼
                            ranking / retrieved evidence / citations
                                                               ▼
                     extractive fallback OR optional local LLM
                                                               ▼
                             Gradio interactive demo

Human-reviewed training data ─> QLoRA optional on CUDA
Independent reviewed gold ─> retrieval evaluation (Recall@K, MRR)
```

Graph associations are text-extracted and should not be described as independently verified knowledge-graph facts.
