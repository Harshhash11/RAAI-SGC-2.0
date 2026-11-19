#!/usr/bin/env python3
"""Independent deterministic structural and source-evidence gate."""
import csv,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'data'/'processed'/'master.csv'
REQUIRED={'record_id','record_type','source_id','source_text','text_hash','source_url','source_revision','retrieved_at','license_status','review_status','split','domain','evidence_quote','edge_weight','tfidf_score','markov_probability','target_concept','target_surface','evidence_start','evidence_end','answer_draft','question','question_id','question_type'}

def validate(path):
 with Path(path).open(encoding='utf8',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames or [];rows=list(r)
 errors=[];warnings=[]; missing=REQUIRED-set(fields)
 if missing:errors.append('Missing schema columns: '+str(sorted(missing)));return {'status':'FAIL','critical_errors':errors}
 counts=Counter(row['record_type'] for row in rows)
 sources={row['source_id']:row for row in rows if row['record_type']=='SOURCE'}
 if len(sources)!=counts['SOURCE']:errors.append('Duplicate source IDs')
 ids=set();qas=set();edges=defaultdict(list);splits=defaultdict(set);bad_hash=0;missing_prov=0;bad_offsets=0;bad_math=0;bad_qa=0;self_loops=0
 for i,x in enumerate(rows,2):
  rid=x['record_id'];sid=x['source_id'];t=x['record_type']
  if rid in ids:errors.append(f'Duplicate record ID line {i}')
  ids.add(rid)
  if t not in {'SOURCE','EDGE','QA_DRAFT'}:errors.append(f'Unknown record type line {i}')
  if sid not in sources:errors.append(f'Orphan record line {i}')
  splits[sid].add(x['split'])
  if hashlib.sha256(x['source_text'].encode('utf8')).hexdigest()!=x['text_hash']:bad_hash+=1
  if not x['source_url'].startswith('https://') or not x['source_revision'] or not x['retrieved_at']:missing_prov+=1
  if t=='EDGE':
   edges[sid].append(x)
   try:
    a,b=int(x['evidence_start']),int(x['evidence_end']);p=float(x['markov_probability']);tfidf=float(x['tfidf_score']);w=float(x['edge_weight']);e=x['evidence_quote']
    if not (0<=a<b<=len(x['source_text']) and x['source_text'][a:b]==e and x['target_surface'].casefold() in e.casefold()):bad_offsets+=1
    if not (0<p<=1 and 0<=tfidf<=1 and math.isclose(w,p*(1+tfidf),rel_tol=1e-9,abs_tol=1e-8)):bad_math+=1
    if x['target_concept'].casefold()==x['concept'].casefold():self_loops+=1
   except (ValueError,OverflowError):bad_math+=1
  if t=='QA_DRAFT':
   if x['question_id'] in qas:errors.append(f'Duplicate QA ID line {i}')
   qas.add(x['question_id'])
   if not x['question'] or not x['evidence_quote'] or x['evidence_quote'] not in x['source_text'] or x['answer_draft'] not in x['evidence_quote']:bad_qa+=1
 for sid,sp in splits.items():
  if len(sp)!=1:errors.append('Source split conflict: '+sid)
 for sid,group in edges.items():
  if not math.isclose(sum(float(x['markov_probability']) for x in group),1,rel_tol=0,abs_tol=1e-8):errors.append('Probability normalization failure: '+sid)
 for k,v in [('Source hashes',bad_hash),('Provenance fields',missing_prov),('Edge evidence offsets',bad_offsets),('Graph formula',bad_math),('Q&A evidence',bad_qa),('Self loops',self_loops)]:
  if v:errors.append(f'{k}: {v} failures')
 if len(sources)<200:warnings.append('Limited IT source coverage; 4 domains have fewer than 200 combined sources')
 if counts['QA_DRAFT']<500:warnings.append('Fewer than 500 draft Q&A records')
 warnings.append('No human source, licensing, or answer approval recorded; cannot certify fine-tuning readiness')
 warnings.append('No independent, human-reviewed gold evaluation benchmark')
 return {'status':'FAIL' if errors else 'CONDITIONAL_PASS','row_count':len(rows),'column_count':len(fields),'record_types':dict(counts),'source_count':len(sources),'domain_source_counts':dict(Counter(x['domain'] for x in sources.values())),'split_source_counts':dict(Counter(x['split'] for x in sources.values())),'critical_errors':errors,'warnings':warnings,'checks':{'record_ids_unique':len(ids)==len(rows),'sources_unique':len(sources)==counts['SOURCE'],'broken_source_links':sum(x['source_id'] not in sources for x in rows),'hash_failures':bad_hash,'provenance_failures':missing_prov,'evidence_offset_failures':bad_offsets,'graph_arithmetic_failures':bad_math,'qa_evidence_failures':bad_qa,'self_loops':self_loops,'transition_sum_failures':sum(not math.isclose(sum(float(x['markov_probability']) for x in group),1,rel_tol=0,abs_tol=1e-8) for group in edges.values())}}

if __name__=='__main__':
 rep=validate(sys.argv[1] if len(sys.argv)>1 else MAIN);print(json.dumps(rep,indent=2));raise SystemExit(bool(rep['critical_errors']))
