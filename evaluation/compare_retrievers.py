"""Three-way retrieval ablation on *independent approved* evaluation questions.

Refuses to create accuracy metrics when no gold review is available.
"""
import argparse, csv, json, time
from pathlib import Path
from src.hybrid import HybridRetriever,load_data

MODES=('vector','unweighted','hybrid')

def approved_gold(path,sources):
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    selected=[r for r in rows if r.get('review_status','').strip().upper()=='APPROVED'
              and r.get('question','').strip() and r.get('evidence_source_id','').strip()]
    if not selected:
        raise ValueError('No APPROVED independent benchmark rows; refusing to report research performance.')
    if any(r['evidence_source_id'] not in sources for r in selected):
        raise ValueError('Approved benchmark references an unknown source ID.')
    if any(r.get('split','').strip() not in ('test','validation') for r in selected):
        raise ValueError('Gold must use validation/test; keep training examples separate.')
    return selected

def compare(master,gold,backend='tfidf',k=5,alpha=.25):
    sources,edges,_=load_data(master)
    selected=approved_gold(gold,sources)
    retriever=HybridRetriever(sources,edges,backend=backend)
    metrics={m:{'hits':0,'rr':0.0,'latencies_ms':[],'graph_activated':0} for m in MODES}
    rows=[]
    for item in selected:
        for mode in MODES:
            begin=time.perf_counter()
            found,diag=retriever.search(item['question'],mode=mode,top_k=k,alpha=alpha)
            elapsed=(time.perf_counter()-begin)*1000
            ids=[r['source_id'] for r in found]
            rank=ids.index(item['evidence_source_id'])+1 if item['evidence_source_id'] in ids else None
            m=metrics[mode]
            m['hits']+=int(rank is not None);m['rr']+=1/rank if rank else 0
            m['latencies_ms'].append(elapsed)
            m['graph_activated']+=int(diag['graph_activated'])
            rows.append({'evaluation_id':item['evaluation_id'],'mode':mode,'rank':rank,
                         'top_source':ids[0] if ids else '', 'elapsed_ms':round(elapsed,3)})
    n=len(selected)
    summary={mode:{'n':n,'recall_at_k':round(m['hits']/n,6),
                   'mrr_at_k':round(m['rr']/n,6),
                   'mean_retrieval_ms':round(sum(m['latencies_ms'])/n,3),
                   'graph_activation_fraction':round(m['graph_activated']/n,6)}
             for mode,m in metrics.items()}
    return {'method':'independent manually APPROVED evidence-source ranking',
            'backend':backend,'top_k':k,'alpha':alpha,'scores':summary,'per_query':rows,
            'caveats':['Gold source ID alone cannot assess answer quality.',
                       'Document retrieval is indexed across all SOURCE records; evaluation labels remain held out.']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master',default='data/processed/master.csv')
    p.add_argument('--gold',default='data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv')
    p.add_argument('--backend',choices=['tfidf','semantic'],default='tfidf')
    p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--alpha',type=float,default=.25)
    p.add_argument('--out',default='reports/generated/three_way_ablation.json')
    a=p.parse_args()
    data=compare(a.master,a.gold,a.backend,a.top_k,a.alpha)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2),encoding='utf8')
    print(json.dumps(data['scores'],indent=2))
if __name__=='__main__':main()
