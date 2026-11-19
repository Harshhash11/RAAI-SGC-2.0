# RAAI-SGC 2.0 — Colab EDA & ML (original university structure retained)

This is ONE project directory. Upload `RAAI_SGC_2_COMPLETE_PROJECT/` itself to the **root of Google Drive > My Drive**. Do not upload only the notebooks: they use `data/processed/master.csv` in the same folder. Do not create a second nested `RAAI_SGC_2_COMPLETE_PROJECT/RAAI_SGC_2_COMPLETE_PROJECT` directory.

## Notebook order

1. `notebooks/01_RAAI_SGC_EDA_COLAB.ipynb`: university EDA template adapted to the real dataset; all its original sections, 15 chart slots, and written prompts are retained. It reads the original master CSV without modifying it.
2. `notebooks/02_RAAI_SGC_ML_COLAB.ipynb`: university ML template adapted to a defensible auxiliary **IT source-domain classification** task. It trains baseline, ComplementNB and LogisticRegression text classifiers, uses training-only cross-validation, selects on validation, assesses held-out test data **once**, visualizes performance, and saves a small classifier to `models/domain_classifier.joblib` and metrics to `reports/generated/`.
3. `notebooks/05_colab_t4_llm_qlora.ipynb`: existing, separate GPU-backed generative LLM / gated QLoRA notebook. It is **not** the same as the supervised domain-classification ML notebook.

The original instructor templates are kept unchanged as `01_university_EDA_TEMPLATE.ipynb` and `02_university_ML_TEMPLATE.ipynb` for direct comparison. The adapted notebooks follow their original headings, question prompts, and 15 chart sections.

## Open in Google Colab

1. Upload/extract the complete project folder to Google Drive root as `MyDrive/RAAI_SGC_2_COMPLETE_PROJECT`.
2. In Drive, locate `notebooks/01_RAAI_SGC_EDA_COLAB.ipynb`; choose **Open with > Google Colaboratory** (or use Colab > File > Open notebook > Google Drive).
3. In Colab, **Runtime > Change runtime type**. For EDA/classical ML a CPU is sufficient; T4 is not required. The QLoRA notebook requires a GPU.
4. **Runtime > Run all**. When prompted, authorize your own Google Drive mount. The notebook automatically finds `MyDrive/RAAI_SGC_2_COMPLETE_PROJECT/data/processed/master.csv`. It raises a clear error if missing.
5. Run the ML notebook only after the EDA notebook has completed; reports are saved under `reports/generated` on Drive.
6. To work locally in Ubuntu, open the *same project folder* in VS Code/Jupyter and run the two notebooks. They also search the current working directory and its parent.

### Minimal Colab package requirements

Colab typically includes pandas, NumPy, matplotlib, SciPy, joblib and scikit-learn. If a required import fails, run:

```python
%pip install -q pandas numpy matplotlib scipy scikit-learn joblib
```

### Dataset contract

`data/processed/master.csv` consists of 5,206 heterogeneous records: 145 `SOURCE`, 4,771 `EDGE`, and 290 `QA_DRAFT`. Treat **SOURCE** records as the 145 independent units for EDA and domain classification. `domain` is the predefined technical subdomain; `source_text` is the input; `split` specifies the fixed source-level train/validation/test partitions. `EDGE` and `QA_DRAFT` are **not** additional independent classification examples. Never train on reserved test examples or unapproved Q&A drafts.

The domain-classification experiment is intended to demonstrate reproducible ML workflow and cannot substitute for LLM fine-tuning, a source-verified question-answering benchmark, or proof that weighted GraphRAG improves answer quality. Source licenses and draft Q&A still require independent review.

### If Colab cannot find the data

Run `!ls /content/drive/MyDrive/RAAI_SGC_2_COMPLETE_PROJECT/data/processed/` and confirm `master.csv` exists. If your folder is nested, move it to MyDrive root or modify `CANDIDATES` in the bootstrap cell. Do not change the CSV schema or move the file out of `data/processed/`.

### Submission details

The university template's original EDA/ML headings and question-answer prompts are preserved in the adapted notebooks. Add your genuine team details, GitHub link, personally written interpretations based on the runtime plots/metrics, and the institution's required declaration. **The generated text is working methodology guidance, not a replacement for original student observations.**
