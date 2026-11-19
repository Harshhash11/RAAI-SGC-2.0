# Interactive Markov–TF-IDF graph (RAAI-SGC 2.0)

## Install on Ubuntu without disturbing your working LLM

```bash
cd ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT_READY
source .venv-llm/bin/activate
python -m pip install -r requirements-graph.txt
python -m pytest tests/test_graph_viz.py -q
python -m app.gradio_app
```

## Use

1. Open http://127.0.0.1:7860.
2. Ask a question such as `What is Kubernetes?` (select hybrid/tfidf).
3. Open **Interactive knowledge graph** tab. It is also populated on initial page load.
4. Pan/zoom the chart, hover over an edge midpoint for its actual Markov probability, TF-IDF score, and final weight, or hover a node for concept/source data.
5. Select **Update graph only (no GPU)** to explore without running the LLM again.
6. Click source URLs listed below the chart to visit the actual Wikipedia pages.
7. Adjust the maximum number of displayed edges to explore additional associations.

## What the graph represents

- **Colored circles**: original source articles, colored by IT domain.
- **Gray diamonds**: concept terms recorded in EDGE records.
- **Lines**: source → concept associations from `data/processed/master.csv`.
- **Line width**: visual encoding of the stored Markov–TF-IDF weight. The tooltip shows exact values; absolute values are small.
- **Blue/orange/green/purple circles**: actual dataset domains rather than a model-predicted classification.

Weight formula: `edge_weight = markov_probability × (1 + tfidf_score)`.

## Scientific limitations

The graph is *not* a semantic ontology, a causal relationship graph, or evidence that all linked entities are factually related. Its edges are source/term co-occurrence associations generated from the dataset; their licenses/review status still require review. A question-specific view shows only the bounded subset of source→term edges for retrieved source articles, not all 4,771 edges. Graph activation is an algorithmic diagnostic, not an independent evaluation of relevance or correctness.

The visualizer does not download a model, use the GPU, or modify training data.
