"""Interactive visualization of *observed* RAAI-SGC SOURCE -> term edges.

The retrieval graph is not a causal KG: edges point from a Wikipedia source to
terms extracted from its text, and weights follow the dataset's Markov--TF-IDF
construction.  No synthetic inter-term edges are drawn.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from src.hybrid import canonical

COLORS = {
    'ai_ml': '#7351C2',
    'cloud_devops': '#1673B8',
    'cybersecurity': '#D65A39',
    'software_engineering': '#23856D',
}

@lru_cache(maxsize=4)
def load_graph_attributes(csv_filename: str):
    """Load actual Markov/TF-IDF components for hover diagnostics."""
    out = defaultdict(dict)
    with Path(csv_filename).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['record_type'] != 'EDGE':
                continue
            key = canonical(row['target_concept'])
            info = {
                'term': row['target_concept'],
                'markov': float(row['markov_probability']),
                'tfidf': float(row['tfidf_score']),
                'weight': float(row['edge_weight']),
            }
            if key not in out[row['source_id']] or info['weight'] > out[row['source_id']][key]['weight']:
                out[row['source_id']][key] = info
    return out


def build_subgraph(sources, edges, results, *, max_edges=36, csv_path=None):
    """Return a bounded, reproducible visualization dataset for retrieved sources.

    Never create a concept-to-concept edge; exact source-title matches are kept
    as labels in edge metadata, not invented semantic associations.
    """
    max_edges = max(1, min(int(max_edges), 120))
    if not results:
        return {'nodes': [], 'edges': [], 'source_ids': []}
    selected = []
    for r in results:
        sid = r['source_id']
        if sid in sources and sid not in selected:
            selected.append(sid)
    attrs = load_graph_attributes(str(Path(csv_path).resolve())) if csv_path else {}
    source_title_index = defaultdict(list)
    for sid, row in sources.items():
        source_title_index[canonical(row['source_title'])].append(sid)
    forced = set()
    for r in results:
        for item in r.get('graph_paths', []):
            forced.add((item.get('from_source'), canonical(item.get('target_term', ''))))
    pool = []
    seen = set()
    for sid in selected:
        for term, weight in edges.get(sid, []):
            key = (sid, canonical(term))
            if not key[1] or key in seen:
                continue
            seen.add(key)
            info = attrs.get(sid, {}).get(key[1], {})
            pool.append({
                'source_id': sid, 'term': term, 'canonical_term': key[1],
                'weight': float(weight),
                'markov': info.get('markov'), 'tfidf': info.get('tfidf'),
                'matches_source_ids': source_title_index.get(key[1], []),
                'forced': key in forced,
            })
    pool.sort(key=lambda e: (not e['forced'], -e['weight'], e['source_id'], e['canonical_term']))
    chosen = pool[:max_edges]
    nodes = []
    for sid in selected:
        s = sources[sid]
        nodes.append({'id': 'source:' + sid, 'label': s['source_title'], 'kind': 'source',
                      'domain': s['domain'], 'url': s['source_url'], 'source_id': sid})
    known = {node['id'] for node in nodes}
    for e in chosen:
        nid = 'term:' + e['canonical_term']
        if nid not in known:
            nodes.append({'id': nid, 'label': e['term'], 'kind': 'concept',
                          'domain': sources[e['source_id']]['domain'], 'url': ''})
            known.add(nid)
        e['from'] = 'source:' + e['source_id']
        e['to'] = nid
    return {'nodes': nodes, 'edges': chosen, 'source_ids': selected}


def make_figure(graph, *, title='Retrieved Markov–TF-IDF knowledge graph'):
    """Create a pan/zoom/hover Plotly network chart; only real CSV edges."""
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError('Install graph viewer packages: python -m pip install -r requirements-graph.txt') from exc
    fig = go.Figure()
    if not graph['nodes']:
        fig.add_annotation(text='No retrieved evidence to graph', showarrow=False, font={'size': 16})
        fig.update_layout(title=title, xaxis={'visible':False}, yaxis={'visible':False})
        return fig
    g = nx.Graph()
    for n in graph['nodes']:
        g.add_node(n['id'])
    for edge in graph['edges']:
        g.add_edge(edge['from'], edge['to'])
    positions = nx.spring_layout(g, seed=23, iterations=85)
    maxweight = max((e['weight'] for e in graph['edges']), default=1.0) or 1.0
    for edge in graph['edges']:
        a, b = positions[edge['from']], positions[edge['to']]
        share = edge['weight'] / maxweight
        fig.add_trace(go.Scatter(
            x=[a[0], b[0]], y=[a[1], b[1]], mode='lines',
            line={'width': 0.9 + 4.0 * share, 'color': f'rgba(65,100,140,{0.25 + 0.65*share:.3f})'},
            hoverinfo='skip', showlegend=False,
        ))
    # Edge hover targets at midpoints; line traces themselves are not reliable hit areas.
    mx=[]; my=[]; hovers=[]
    for edge in graph['edges']:
        a,b=positions[edge['from']],positions[edge['to']]
        mx.append((a[0]+b[0])/2); my.append((a[1]+b[1])/2)
        source=next(n for n in graph['nodes'] if n['id']==edge['from'])
        h=f"<b>{source['label']} → {edge['term']}</b><br>Markov–TF-IDF weight: {edge['weight']:.8f}"
        if edge['markov'] is not None:
            h+=f"<br>Markov probability: {edge['markov']:.8f}<br>TF-IDF: {edge['tfidf']:.8f}"
        if edge['matches_source_ids']:
            h+='<br>Exact source-title match present'
        if edge['forced']:
            h+='<br>Used in retrieval expansion'
        hovers.append(h)
    if mx:
        fig.add_trace(go.Scatter(x=mx, y=my, mode='markers',
                                 marker={'size':14,'color':'rgba(0,0,0,0)'},
                                 hovertext=hovers, hoverinfo='text', showlegend=False))
    for domain, color in COLORS.items():
        subset=[n for n in graph['nodes'] if n['kind']=='source' and n['domain']==domain]
        if not subset: continue
        fig.add_trace(go.Scatter(
            x=[positions[n['id']][0] for n in subset],
            y=[positions[n['id']][1] for n in subset], mode='markers+text',
            text=[n['label'] for n in subset], textposition='top center',
            textfont={'size':12},
            marker={'size':23,'color':color,'line':{'width':2,'color':'white'}},
            hovertext=[f"<b>{n['label']}</b><br>Domain: {domain}<br>Source URL: {n['url']}" for n in subset],
            hoverinfo='text', name=f'Source · {domain.replace("_", " / ")}',
        ))
    terms=[n for n in graph['nodes'] if n['kind']=='concept']
    if terms:
        fig.add_trace(go.Scatter(
            x=[positions[n['id']][0] for n in terms],
            y=[positions[n['id']][1] for n in terms],
            mode='markers+text', text=[n['label'] for n in terms],
            textposition='bottom center', textfont={'size':10},
            marker={'size':11,'color':'#94A3B8','symbol':'diamond', 'line':{'width':1,'color':'white'}},
            hovertext=[f"<b>Extracted term:</b> {n['label']}" for n in terms],
            hoverinfo='text', name='Extracted concept (not verified entity)',
        ))
    fig.update_layout(
        title=title, height=650, template='plotly_white',
        margin={'l':15,'r':15,'t':55,'b':15},
        hovermode='closest',
        legend={'orientation':'h','yanchor':'bottom','y':1.02,'x':0},
        xaxis={'visible':False},yaxis={'visible':False, 'scaleanchor':'x','scaleratio':1},
        dragmode='pan',
    )
    return fig


def source_links(graph, sources):
    """Clickable grounded URLs alongside the Plotly chart."""
    if not graph['source_ids']:
        return 'No matching source articles.'
    lines = [f"**Graph content:** {len(graph['source_ids'])} retrieved source(s), "
             f"{len(graph['edges'])} selected dataset edge(s). Hover nodes/edges for details; drag to pan, scroll to zoom.",
             '', '**Clickable Wikipedia source articles**']
    for sid in graph['source_ids']:
        s=sources[sid]
        if s['source_url'].startswith(('https://','http://')):
            lines.append(f"- [{s['source_title']}]({s['source_url']}) — `{s['domain']}`")
    lines.append('')
    lines.append('**Interpretation:** Source→term edges are extracted associations, not proven causal or semantic relationships. '
                 'Edge weights are from the master CSV; the graph is a bounded view, not all 4,771 edges.')
    return '\n'.join(lines)
