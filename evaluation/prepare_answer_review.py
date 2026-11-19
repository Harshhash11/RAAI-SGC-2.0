"""Create answer-quality review sheets from REAL recorded LLM generations.

Does not score answers automatically and does not call unapproved labels gold.
"""
import argparse,csv,json
from pathlib import Path
FIELDS=['evaluation_id','domain','question','retrieval_mode','answer','retrieved_source_ids',
        'factual_correctness_1_to_5','groundedness_1_to_5','completeness_1_to_5',
        'citation_accuracy_1_to_5','reviewer','review_date','review_notes']

def create(predictions,out):
    with open(predictions,encoding='utf8') as f: data=json.load(f)
    if not isinstance(data,list): raise ValueError('Expected a list of real model predictions')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for i,x in enumerate(data,1):
            if not x.get('answer') or not x.get('question'): continue
            writer.writerow({'evaluation_id':x.get('evaluation_id',str(i)),
              'domain':x.get('domain',''),'question':x['question'],
              'retrieval_mode':x.get('retrieval_mode',''), 'answer':x['answer'],
              'retrieved_source_ids':';'.join(x.get('source_ids',[]))})
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',default='reports/generated/llm_answer_human_review.csv');a=p.parse_args();print(create(a.predictions,a.out))
