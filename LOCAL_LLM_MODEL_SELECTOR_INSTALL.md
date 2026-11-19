# RAAI-SGC 2.0 — Local LLM model selector update

This patch upgrades the **graph-viewer version** of `app/gradio_app.py`. It adds a Local LLM model dropdown; `Qwen/Qwen2.5-1.5B-Instruct` is preselected. The selector is hidden in Extractive (offline) mode. Choose **Custom model / local adapter** to enter a different Hugging Face model ID or the directory of a fine-tuned LoRA adapter. No modification is made to `src/llm.py`, `.venv-llm`, model files, datasets or research notebooks.

## Install

Stop Gradio with Ctrl+C, then run in Ubuntu:

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
cp app/gradio_app.py app/gradio_app.py.before_model_selector
unzip -o ~/Downloads/RAAI_SGC_LOCAL_LLM_SELECTOR_UPDATE.zip -d .
source .venv-llm/bin/activate
python -m pytest tests/ -q
python -m app.gradio_app
```

This patch assumes `RAAI_SGC_GRAPH_VIEWER_UPDATE.zip` was already installed (including `src/graph_viz.py` and its plotting dependencies). If not, install that graph viewer update first. Do **not** launch via `./start.sh chat`, which may select `.venv` instead of `.venv-llm`.

## Use

- Select `Extractive (offline)` for source-based offline replies; model selection is hidden.
- Select `Local LLM` to reveal the model menu. The default is `Qwen/Qwen2.5-1.5B-Instruct`.
- Select `Custom model / local adapter` only if you wish to enter a different model ID or a saved adapter path.
- Select hybrid retrieval and TF-IDF backend; click `Ask + update graph` to generate an answer and update the graph. The graph tab can also be refreshed without GPU inference.

The code and app HTTP launch were tested here; actual RTX 3060 inference remains verified by the user's separate working project and must be retested after installing this UI-only patch.
