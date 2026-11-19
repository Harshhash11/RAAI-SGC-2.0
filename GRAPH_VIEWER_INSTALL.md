# RAAI-SGC 2.0 — Interactive graph viewer in-place update

This is **not** a separate project. Unzip its contents **into** your existing
`~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY` directory.

## Ubuntu commands

Stop the running Gradio server first with Ctrl+C, then:

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
cp app/gradio_app.py app/gradio_app.py.before_graph
unzip -o ~/Downloads/RAAI_SGC_GRAPH_VIEWER_UPDATE.zip -d .
source .venv-llm/bin/activate
python -m pip install -r requirements-graph.txt
python -m pytest tests/ -q
python -m app.gradio_app
```

Visit http://127.0.0.1:7860, then select **Interactive knowledge graph**.
Try the `hybrid` / `tfidf` retrieval mode and `What is Kubernetes?`.
The separate **Update graph only (no GPU)** button updates the diagram without invoking Qwen.

**Do not run `./start.sh chat`:** the current start script uses `.venv`, not `.venv-llm`.
The patch leaves `src/llm.py`, your model cache, existing training data, and notebooks intact.

## Change summary

- `app/gradio_app.py` — extends the existing UI with a new interactive graph tab
- `src/graph_viz.py` — draws source→term weighted associations from the real dataset
- `requirements-graph.txt` — Plotly/NetworkX viewer dependencies
- `tests/test_graph_viz.py` — validation of displayed edges and weight arithmetic
- `docs/GRAPH_VIEWER.md` — usage and research limitations

The viewer includes node color legends, hover details, zoom/pan, and clickable
Wikipedia source links beneath the graph. Concept nodes are unreviewed extracted
terms, not certified semantic concepts.
