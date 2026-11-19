"""Gradio RAAI-SGC chatbot with source-grounded interactive graph viewer."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hybrid import load_data, HybridRetriever
from src.answers import extractive_answer
from src.llm import generate
from src.graph_viz import build_subgraph, make_figure, source_links

_RETRIEVERS = {}
DEFAULT_LOCAL_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
CUSTOM_MODEL = 'Custom model / local adapter'
FINE_TUNED_MODEL = 'Fine-tuned Qwen (RAAI-SGC)'
FINE_TUNED_ADAPTER = ROOT / 'models/checkpoints/qlora_final_44'


def resolve_local_model(response_type, choice, custom_model=''):
    """Return the model only for Local LLM requests; never influence extractive mode."""
    if response_type != 'Local LLM':
        return ''
    if choice == CUSTOM_MODEL:
        if not custom_model.strip():
            raise ValueError('Enter a Hugging Face model ID or saved adapter directory.')
        return custom_model.strip()
    if choice == DEFAULT_LOCAL_MODEL:
        return DEFAULT_LOCAL_MODEL
    if choice == FINE_TUNED_MODEL:
        if not (FINE_TUNED_ADAPTER / 'adapter_config.json').is_file():
            raise ValueError('Fine-tuned adapter missing: ' + str(FINE_TUNED_ADAPTER) + '. Select pretrained Qwen or check the adapter path.')
        return str(FINE_TUNED_ADAPTER)
    raise ValueError('Choose a supported model or Custom model / local adapter.')


def model_selector_visibility(response_type):
    """UI visibility for models: local mode only."""
    is_local = response_type == 'Local LLM'
    return is_local


def get_retriever(backend):
    if backend not in _RETRIEVERS:
        sources, edges, _ = load_data(ROOT / 'data/processed/master.csv')
        _RETRIEVERS[backend] = HybridRetriever(sources, edges, backend=backend)
    return _RETRIEVERS[backend]


def retrieved_evidence_links(results):
    """Show retriever provenance without falsely claiming the LLM cited it."""
    links = []
    seen = set()
    for i, hit in enumerate(results, 1):
        url = str(hit.get('source_url') or '').strip()
        title = str(hit.get('source_title') or 'Untitled source').replace('[', '').replace(']', '')
        if not url.startswith(('https://', 'http://')) or url in seen:
            continue
        seen.add(url)
        # Escape parentheses to prevent broken Markdown links.
        safe_url = url.replace('(', '%28').replace(')', '%29')
        links.append(f"{i}. [{title}]({safe_url})")
    return '\n'.join(links)


def format_answer_with_evidence(response, results):
    """Distinguish explicit model citations from merely retrieved evidence."""
    message = str(response.get('answer') or '').strip()
    citations = response.get('citations') or []
    if citations:
        lines = []
        for source in citations:
            url = str(source.get('url') or '')
            title = str(source.get('title') or 'Source')
            if url.startswith(('http://', 'https://')):
                lines.append(f"- [{title}]({url.replace('(', '%28').replace(')', '%29')})")
        if lines:
            message += '\n\n**Sources explicitly cited in the answer**\n' + '\n'.join(lines)
    links = retrieved_evidence_links(results)
    if links:
        message += '\n\n**Retrieved source documents (not automatically verified as supporting every claim)**\n' + links
    elif not citations:
        message += '\n\n_No source documents available for this answer._'
    return message


def answer(question, mode='hybrid', backend='tfidf', response_type='Extractive (offline)', model_name='', custom_model=''):
    if not question.strip():
        return 'Please enter a question.', 'No query submitted.'
    try:
        results, diagnostic = get_retriever(backend).search(question, top_k=3, mode=mode)
        if response_type == 'Local LLM':
            response = generate(question, results, resolve_local_model(response_type, model_name, custom_model))
        else:
            response = extractive_answer(question, results)
        message = format_answer_with_evidence(response, results)
        details = (f"Retrieval: {mode} / {backend}\nGraph activated: {diagnostic['graph_activated']}\n" +
                   '\n'.join(f"{i+1}. {r['source_title']} (vector={r['vector_score']:.3f}, graph={r['graph_score']:.3f})" for i,r in enumerate(results)))
        return message, details
    except Exception as exc:
        return f'Could not process query: {type(exc).__name__}: {exc}', 'Check optional dependencies or model access.'


def graph_view(question, mode='hybrid', backend='tfidf', max_edges=36):
    """Graph rendering never invokes the LLM and therefore works offline/on CPU."""
    retriever = get_retriever(backend)
    hits, diagnostic = retriever.search(question, top_k=3, mode=mode)
    graph = build_subgraph(retriever.sources, retriever.edges, hits,
                           max_edges=max_edges, csv_path=ROOT/'data/processed/master.csv')
    chart = make_figure(graph, title=f'Observed associations for: {question[:72]}')
    notes = source_links(graph, retriever.sources)
    if diagnostic['graph_activated']:
        notes = '**Weighted retrieval expansion activated for this query.**\n\n' + notes
    return chart, notes


def ask_with_graph(question, mode, backend, response_type, model_name, custom_model, max_edges):
    message, details = answer(question, mode, backend, response_type, model_name, custom_model)
    try:
        chart, notes = graph_view(question, mode, backend, max_edges)
    except Exception as exc:
        chart = None
        notes = f'Graph viewer error: {type(exc).__name__}: {exc}'
    return message, details, chart, notes


def main():
    import gradio as gr
    with gr.Blocks(title='RAAI-SGC IT Knowledge Assistant') as demo:
        gr.Markdown('# RAAI-SGC 2.0 — IT Knowledge Assistant\n**Local industry capstone demo.** Retrieval sources are shown separately from model-confirmed citations; generated statements require verification.')
        question = gr.Textbox(label='Your IT question', value='What is Kubernetes?')
        with gr.Row():
            mode = gr.Dropdown(['vector','unweighted','hybrid'], value='hybrid', label='Retrieval method')
            backend = gr.Dropdown(['tfidf','semantic'], value='tfidf', label='Retrieval backend')
        with gr.Row():
            response_type = gr.Dropdown(['Extractive (offline)','Local LLM'],value='Extractive (offline)',label='Response method')
            model_name = gr.Dropdown(
                choices=[DEFAULT_LOCAL_MODEL, FINE_TUNED_MODEL, CUSTOM_MODEL],
                value=DEFAULT_LOCAL_MODEL,
                label='Local LLM model', visible=False, interactive=True
            )
        custom_model = gr.Textbox(
            label='Custom Hugging Face model ID or saved adapter directory',
            value=os.environ.get('RAAI_LLM_MODEL', 'models/checkpoints/qlora_final_44'), visible=False
        )
        response_type.change(
            lambda choice, response: (gr.update(visible=model_selector_visibility(choice)),
                                      gr.update(visible=model_selector_visibility(choice) and response == CUSTOM_MODEL)),
            [response_type, model_name], [model_name, custom_model]
        )
        model_name.change(
            lambda choice, response: gr.update(visible=model_selector_visibility(response) and choice == CUSTOM_MODEL),
            [model_name, response_type], [custom_model]
        )
        gr.Markdown('**Tip:** Choose *Fine-tuned Qwen (RAAI-SGC)* under Local LLM to load your saved QLoRA adapter. On a 6 GB GPU, run only one Gradio server at a time.')
        with gr.Row():
            run = gr.Button('Ask + update graph', variant='primary')
            graph_refresh = gr.Button('Update graph only (no GPU)')
        with gr.Tabs():
            with gr.Tab('Answer & diagnostics'):
                answer_md = gr.Markdown()
                details = gr.Textbox(label='Retrieval diagnostics', lines=5)
            with gr.Tab('Interactive knowledge graph'):
                gr.Markdown('**Real master.csv edges:** source pages → extracted concepts. Zoom with mouse wheel, drag to pan, hover to inspect exact weights. Source links are listed below.')
                max_edges = gr.Slider(8, 100, value=36, step=4, label='Maximum displayed weighted edges')
                graph_plot = gr.Plot(label='Graph of retrieved evidence')
                graph_notes = gr.Markdown()
        run.click(ask_with_graph,
                  [question, mode, backend, response_type, model_name, custom_model, max_edges],
                  [answer_md, details, graph_plot, graph_notes])
        graph_refresh.click(graph_view,
                            [question, mode, backend, max_edges],
                            [graph_plot, graph_notes])
        demo.load(graph_view, [question, mode, backend, max_edges], [graph_plot, graph_notes])
    demo.launch(server_name='127.0.0.1')


if __name__=='__main__':
    main()
