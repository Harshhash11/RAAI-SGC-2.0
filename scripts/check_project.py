"""One command to audit project completeness without downloading LLM weights."""
import csv,json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hybrid import load_data,HybridRetriever

def main():
    required=['data/processed/master.csv','app/gradio_app.py','src/hybrid.py',
              'training/train_qlora_colab.py','notebooks/05_colab_t4_llm_qlora.ipynb',
              'notebooks/06_colab_drive_inference_and_training.ipynb',
              'data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv']
    missing=[x for x in required if not (ROOT/x).is_file()]
    if missing: raise RuntimeError('Missing project files: '+repr(missing))
    sources,edges,drafts=load_data(ROOT/'data/processed/master.csv')
    with (ROOT/'data/training/sft_train.jsonl').open(encoding='utf-8') as f:
        examples=sum(bool(s.strip()) for s in f)
    with (ROOT/'data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv').open(encoding='utf-8-sig') as f:
        gold=sum(r.get('review_status','').upper()=='APPROVED' for r in csv.DictReader(f))
    r=HybridRetriever(sources,edges)
    hits,diag=r.search('What is Kubernetes?',mode='hybrid',top_k=3)
    assert hits and hits[0]['source_title']=='Kubernetes'
    report={'sources':len(sources),'edges':sum(len(v) for v in edges.values()),
            'draft_qa':drafts,'approved_sft_examples':examples,
            'approved_independent_eval':gold,'kubernetes_top_hit':hits[0]['source_title'],
            'graph_activated_for_smoke_test':diag['graph_activated'],
            'optional_gradio_installed':importlib.util.find_spec('gradio') is not None,
            'optional_transformers_installed':importlib.util.find_spec('transformers') is not None,
            'status':'CORE_CHECKS_PASSED; GPU and review gates still pending'}
    print(json.dumps(report,indent=2))
    out=ROOT/'reports/generated/project_check.json';out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
