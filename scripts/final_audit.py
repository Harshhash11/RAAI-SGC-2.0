#!/usr/bin/env python3
"""Generate an evidence-based project readiness matrix (without invented metrics)."""
import csv,json,hashlib,importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hybrid import load_data
from training.validation import read_csv


def notebook_summary(path):
    import nbformat
    nb=nbformat.read(str(path),as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    return {'cells':len(nb.cells),'code_cells':len(code),'code_with_saved_output':sum(bool(c.get('outputs')) for c in code),
            'blank_cells':sum(not c.source.strip() for c in nb.cells),
            'error_outputs':sum(any(o.output_type=='error' for o in c.get('outputs',[])) for c in code)}


def audit():
    master=ROOT/'data/processed/master.csv'
    sources,edges,drafts=load_data(master)
    sft=ROOT/'data/training/sft_train.jsonl'
    ntrain=sum(bool(x.strip()) for x in sft.read_text(encoding='utf8').splitlines()) if sft.exists() else 0
    gold=ROOT/'data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv'
    ngold=sum(r.get('review_status','').upper()=='APPROVED' for r in read_csv(gold))
    adapter=ROOT/'models/checkpoints/colab_qlora/adapter_config.json'
    llm_run=ROOT/'reports/generated/colab_llm_baseline.json'
    eda=notebook_summary(ROOT/'notebooks/01_RAAI_SGC_EDA_COLAB.ipynb')
    ml=notebook_summary(ROOT/'notebooks/02_RAAI_SGC_ML_COLAB.ipynb')
    return {'dataset':{'source_count':len(sources),'edge_count':sum(len(v) for v in edges.values()),'unapproved_qa_drafts':drafts,'sha256':hashlib.sha256(master.read_bytes()).hexdigest()},
      'notebooks':{'EDA':eda,'ML':ml},
      'model_artifacts':{'trained_domain_classifier':(ROOT/'models/exports/domain_classifier.joblib').exists(),
                         'gpu_qlora_adapter_present':adapter.exists(),'colab_llm_baseline_record_present':llm_run.exists()},
      'human_review':{'supervised_train_rows':ntrain,'independent_approved_eval_rows':ngold},
      'submission_ready':bool(adapter.exists() and llm_run.exists() and ntrain>=20 and ngold>=20 and eda['error_outputs']==ml['error_outputs']==0),
      'limitations':['Cannot infer approval from code','Artifact presence does not certify model quality','Final academic submission needs researcher review, correct citations and evidence of actual GPU runs']}


def main():
    report=audit(); target=ROOT/'reports/generated/final_readiness_audit.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    return 0 if report['dataset']['source_count']==145 and report['notebooks']['EDA']['error_outputs']==report['notebooks']['ML']['error_outputs']==0 else 1
if __name__=='__main__':raise SystemExit(main())
