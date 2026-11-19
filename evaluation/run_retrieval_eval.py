"""Strict retrieval evaluation: only externally reviewed held-out benchmark rows.
Never generate labels from the same source being searched.
"""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hybrid import load_data,HybridRetriever

def evaluate(master,gold,backend='tfidf',mode='vector',top_k=5):
    sources,edges,_=load_data(master)
    with open(gold,newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    selected=[r for r in rows if r.get('review_status','').strip().upper()=='APPROVED' and r.get('question','').strip() and r.get('evidence_source_id','').strip()]
    if not selected:
        raise ValueError('No reviewer-approved evaluation examples. Fill and approve data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv first.')
    if any(r['evidence_source_id'] not in sources for r in selected):
        raise ValueError('Some gold source IDs are absent from master')
    retriever=HybridRetriever(sources,edges,backend)
    hits=0;mrr=0.;details=[];start=time.perf_counter()
    for r in selected:
        results,_=retriever.search(r['question'],top_k=top_k,mode=mode)
        ranked=[x['source_id'] for x in results]
        rank=(ranked.index(r['evidence_source_id'])+1) if r['evidence_source_id'] in ranked else None
        hits+=bool(rank);mrr+=1./rank if rank else 0
        details.append({'evaluation_id':r['evaluation_id'],'rank':rank,'top_result':ranked[0] if ranked else None})
    n=len(selected)
    return {'reviewed_n':n,'backend':backend,'mode':mode,'k':top_k,'recall_at_k':hits/n,'mrr_at_k':mrr/n,'elapsed_seconds':round(time.perf_counter()-start,3),'per_question':details}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master',default='data/processed/master.csv');p.add_argument('--gold',default='data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv')
    p.add_argument('--backend',choices=['tfidf','semantic'],default='tfidf');p.add_argument('--mode',choices=['vector','unweighted','hybrid'],default='vector')
    p.add_argument('--top-k',type=int,default=5);p.add_argument('--out',default='reports/generated/retrieval_eval.json')
    a=p.parse_args();result=evaluate(a.master,a.gold,a.backend,a.mode,a.top_k)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_question'},indent=2))
if __name__=='__main__':main()
