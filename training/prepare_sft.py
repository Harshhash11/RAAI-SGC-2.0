"""Produces JSONL only from reviewed, explicitly approved train-set questions."""
import csv,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def prepare(path,out):
    with open(path,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    approved=[r for r in rows if r.get('record_type')=='QA_DRAFT' and r.get('review_status')=='APPROVED' and r.get('license_status')=='APPROVED' and r.get('split')=='train' and r.get('question','').strip() and r.get('answer_draft','').strip()]
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf8') as f:
        for r in approved:
            obj={'messages':[{'role':'system','content':'You are a helpful IT technical assistant. Answer accurately and do not invent facts.'},{'role':'user','content':r['question']},{'role':'assistant','content':r['answer_draft']}], 'source_id':r['source_id'],'domain':r['domain']}
            f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    return len(approved)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approved',default=str(ROOT/'data/review/APPROVED_QA_ONLY.csv'));p.add_argument('--out',default=str(ROOT/'data/training/sft_train.jsonl'));a=p.parse_args();n=prepare(a.approved,a.out);print(f'Prepared {n} approved training examples. No drafts used.')
