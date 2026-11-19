# Required final actions that cannot be outsourced to a CPU-only packaging run

1. **Source rights/quality:** A real reviewer should check source text, attribution, domain relevance and completeness in `data/review/SOURCE_REVIEW_QUEUE.csv`. Do not bulk-mark approvals.
2. **Q&A review:** Domain reviewers check `data/review/QA_REVIEW_QUEUE.csv` and enter reviewer names, reviewed answers, and criteria. Run `python dataset_engineering/export_approved.py` followed by `python -m training.prepare_sft`.
3. **GPU model run:** Upload the canonical `RAAI_SGC_2_COMPLETE_PROJECT` folder to `MyDrive/`, open `notebooks/06_colab_drive_inference_and_training.ipynb`, choose T4 and execute in order. It saves generation predictions and a blank human-review rubric. If 20+ valid approved rows are present, the notebook attempts a capped QLoRA training run and saves adapter/logs. Actual success depends on runtime resources and package compatibility.
4. **Independent evaluation:** Populate `data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv` with genuinely judged held-out questions and `APPROVED` status. Run `python -m evaluation.compare_retrievers` and compare generated answers using the review CSV.
5. **Academic finalization:** Fill real names, contribution statements, citations, ethics/license declarations, figures, references, final paper in university format. No ready-to-submit claim until the evidence exists.

## Folder policy
The complete main project is `RAAI_SGC_2_COMPLETE_PROJECT`. The EDA/ML-only submission folder remains distinct. Do not delete the master CSV, review records, scripts or test fixtures. Keep an unmodified backup before replacing a working Ubuntu folder.
