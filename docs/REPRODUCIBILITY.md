# Reproducing recorded results

## Ubuntu CPU

```bash
cd RAAI_SGC_2_COMPLETE_PROJECT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-app.txt
python -m pytest -q
python scripts/check_project.py
python -m app.gradio_app
```

### Saved results

- `reports/generated/executed_checks.json`: test output and project checks.
- `reports/generated/retrieval_title_self_probes.json`: title-derived source self-retrieval diagnostic (not a gold benchmark).
- `reports/generated/measured_results.json`: dataset/auxiliary ML experiment summary.
- `reports/generated/ml_domain_classification_metrics.json`: auxiliary domain-classification metrics.
- `notebooks/01_RAAI_SGC_EDA_COLAB.ipynb` and `notebooks/02_RAAI_SGC_ML_COLAB.ipynb`: executed outputs and numerical interpretation.

### Google Colab T4

Put the **same** `RAAI_SGC_2_COMPLETE_PROJECT` folder at `MyDrive/RAAI_SGC_2_COMPLETE_PROJECT`. Use `notebooks/06_colab_drive_inference_and_training.ipynb` with T4 runtime. Do not claim GPU experiment completion until you run it, save outputs and record model version, checkpoint, GPU/runtime packages, prompt config, input labels and actual measurements.
