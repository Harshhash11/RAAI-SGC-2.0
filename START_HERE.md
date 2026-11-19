# START HERE — RAAI-SGC 2.0 (Ubuntu / VS Code)

This is the **main industry project**, not the separate EDA/ML-only submission folder.

## Download/extract correctly — important

This ZIP is **flat**: `requirements.txt`, `start.sh`, `app/`, `data/`, and `notebooks/` are at the ZIP root. Do not extract inside another folder that already contains an older project. On Ubuntu, from Downloads:

```bash
mkdir -p ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT
unzip -o ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY.zip -d ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT
ls requirements.txt start.sh data/processed/master.csv
chmod +x start.sh
./start.sh setup
./start.sh check
./start.sh chat
```

After `chat`, visit <http://127.0.0.1:7860>. Stop with Ctrl+C.

**Using the graphical Archive Manager:** Extract into **one fresh folder**, open that folder in VS Code, and run `./start.sh setup` from its terminal. You must see `requirements.txt` beside `start.sh`.

## What runs where

- Ubuntu CPU: Gradio extractive bot, TF-IDF, weighted/unweighted graph reranking, automated tests, EDA/ML code, saved auxiliary domain classifier.
- Optional Ubuntu download: sentence-transformer embeddings. Run `.venv/bin/python -m pip install -r requirements-semantic.txt` and use semantic backend.
- Google Colab T4: `notebooks/06_colab_drive_inference_and_training.ipynb` for pretrained Qwen LLM and guarded QLoRA.
- Human review: `data/review/QA_REVIEW_QUEUE.csv` and independent gold evaluation template must be approved *before* supervised fine-tuning or final research accuracy claims.

## Project completion status

Tests pass locally and the ML domain classifier is saved. **No genuine QLoRA adapter or reviewed LLM benchmark exists in this package.** The QLoRA notebook and code are included but cannot be honestly executed with zero approved Q&A and without GPU access. See `docs/FINAL_ACTIONS_AND_OWNERSHIP.md`.

## Notes on the notebooks

`notebooks/01_RAAI_SGC_EDA_COLAB.ipynb` and `02_RAAI_SGC_ML_COLAB.ipynb` contain saved outputs and preserve the university-template structure. Their original Google Drive paths may require using Colab to rerun from source. Run and review results for your own submission.
