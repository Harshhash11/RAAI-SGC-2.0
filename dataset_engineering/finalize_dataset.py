#!/usr/bin/env python3
"""Deterministic, offline Part 1 freeze from the user's corrected Wikipedia corpus.
Do not convert automated flags into expert approvals. No external dependencies.
"""
import argparse,csv,hashlib,json,re,statistics
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'input'/'BASELINE_149_SOURCES.csv'
OUT=ROOT/'output'
MAIN=OUT/'RAAI_SGC_2_MASTER_DATASET_PART1.csv'
# Exclusions verified from the exact article titles. Preserve in a separate audit record.
EXCLUDE={
 'Ancient language':'Linguistics/history article outside the four IT subdomains',
 'Arabic':'Natural language article, not a technical NLP-method source',
 'Neural network (biology)':'Biological network article; retain computational neural-network source instead',
 'Software Engineering Notes':'Academic journal entry rather than a technical software-engineering concept',
}
GENERIC={'thing','way','part','term','type','example','case','number','time','year','day','fact','order','group','level','form','area','process','system','method','use','field','problem','role','ability','information','application','service','task','component','tool','function','development','organization','object','action','feature','concept'}
BADPREFIX=re.compile(r'^(?:e\s*\.\s*g\s*\.|i\s*\.\s*e\s*\.)',re.I)
BADCHARS=re.compile(r'[{}<>]|\b(?:\d{1,3},\d{3})\b')

def read_csv(p):
 with p.open(encoding='utf-8',newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)

def write_csv(p,fields,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
 fields,records=read_csv(INPUT)
 original_sources=[r for r in records if r['record_type']=='SOURCE']
 excluded_sources=[r for r in original_sources if r['source_title'] in EXCLUDE]
 exclude_ids={r['source_id'] for r in excluded_sources}
 records=[r.copy() for r in records if r['source_id'] not in exclude_ids]
 for r in records:r['dataset_version']='PART1-1.0-candidate'
 # Excluding whole source components doesn't alter the outgoing weights of other source nodes.
 write_csv(MAIN,fields,records)
 write_csv(OUT/'EXCLUDED_SOURCES.csv',['source_id','source_title','domain','source_url','reason'],[{'source_id':r['source_id'],'source_title':r['source_title'],'domain':r['domain'],'source_url':r['source_url'],'reason':EXCLUDE[r['source_title']]} for r in excluded_sources])
 sources=[r for r in records if r['record_type']=='SOURCE'];edges=[r for r in records if r['record_type']=='EDGE'];qas=[r for r in records if r['record_type']=='QA_DRAFT']
 by_source=Counter(r['source_id'] for r in edges)
 src_rows=[]
 for r in sorted(sources,key=lambda x:(x['domain'],x['source_title'].casefold())):
  text=r['source_text']; flags=[]
  if len(text)<350:flags.append('SHORT_SOURCE_TEXT')
  if text.rstrip().endswith('...'):flags.append('ELLIPSIS_END')
  if len(text.split())<45:flags.append('LOW_WORD_COUNT')
  if 'disambiguation' in r['source_title'].lower():flags.append('POSSIBLE_DISAMBIGUATION')
  src_rows.append({'source_id':r['source_id'],'source_title':r['source_title'],'domain':r['domain'],'split':r['split'],'source_url':r['source_url'],'source_revision':r['source_revision'],'text_characters':len(text),'graph_edges':by_source[r['source_id']],'quality_flags':';'.join(flags),'domain_relevance':'PENDING','factual_accuracy':'PENDING','text_completeness':'PENDING','license_review':'PENDING','attribution_review':'PENDING','reviewer':'','review_date':'','review_notes':''})
 write_csv(OUT/'SOURCE_REVIEW_QUEUE.csv',list(src_rows[0]),src_rows)
 terms=defaultdict(lambda:{'ids':set(),'domains':set(),'n':0,'examples':[]})
 for r in edges:
  t=terms[r['target_concept']];t['ids'].add(r['source_id']);t['domains'].add(r['domain']);t['n']+=1
  if len(t['examples'])<2:t['examples'].append(r['target_surface'])
 term_rows=[]
 for k,v in sorted(terms.items()):
  flags=[]
  if k in GENERIC:flags.append('GENERIC_TERM')
  if BADPREFIX.search(k):flags.append('BAD_PREFIX')
  if BADCHARS.search(k):flags.append('MALFORMED_OR_NUMERIC')
  if k in {'datum','datum mining','datum confidentiality'} or k.startswith('datum '):flags.append('CHECK_LEMMATIZATION')
  term_rows.append({'target_concept':k,'source_count':len(v['ids']),'edge_records':v['n'],'domains':';'.join(sorted(v['domains'])),'surface_examples':';'.join(v['examples']),'quality_flags':';'.join(flags),'review_decision':'PENDING','canonical_override':'','reviewer':'','review_notes':''})
 write_csv(OUT/'TERM_REVIEW_QUEUE.csv',list(term_rows[0]),term_rows)
 qa_rows=[]
 for r in sorted(qas,key=lambda x:(x['source_id'],x['question_type'])):
  flags=[]
  if r['question_type']!='definition':flags.append('TEMPLATE_CONTEXT_QUESTION')
  if len(r['answer_draft'])>500:flags.append('LONG_EXTRACTIVE_ANSWER')
  if len(r['answer_draft'])<80:flags.append('SHORT_ANSWER')
  if not re.search(r'[.!?]["\)\]]?$',r['answer_draft'].strip()):flags.append('POSSIBLE_INCOMPLETE_SENTENCE')
  qa_rows.append({'question_id':r['question_id'],'source_id':r['source_id'],'source_title':r['source_title'],'domain':r['domain'],'split':r['split'],'question_type':r['question_type'],'question':r['question'],'answer_draft':r['answer_draft'],'evidence_quote':r['evidence_quote'],'quality_flags':';'.join(flags),'answer_factuality':'PENDING','answer_relevance':'PENDING','answer_completeness':'PENDING','evidence_adequacy':'PENDING','approved_answer':'','reviewer':'','review_date':'','review_notes':''})
 write_csv(OUT/'QA_REVIEW_QUEUE.csv',list(qa_rows[0]),qa_rows)
 # Dataset dictionary is generated from the actual schema.
 defs={
 'record_id':'Stable unique row identifier','record_type':'SOURCE, EDGE, or QA_DRAFT','dataset_version':'Version of consolidated dataset','domain':'Primary IT domain','secondary_domains':'Optional additional IT domains','source_id':'Stable Wikipedia source identifier','source_title':'Wikipedia page title','source_url':'Link to evidence source','source_revision':'Wikipedia revision identifier','retrieved_at':'UTC source retrieval timestamp','source_provider':'Original information provider','license_status':'Unreviewed licensing status; not permission','review_status':'Human review status; not an automated quality grade','split':'Source-grouped train, validation or test partition','concept_id':'Identifier for page-level concept','concept':'Primary source concept','source_text':'Retrieved article extract; not guaranteed exhaustive','text_hash':'SHA-256 of source_text','target_concept':'Normalized graph target term','target_surface':'Exact surface text extracted from evidence','evidence_start':'Inclusive character offset for EDGE evidence','evidence_end':'Exclusive character offset for EDGE evidence','markov_probability':'Outgoing source transition probability','tfidf_score':'Target TF-IDF score','edge_weight':'Markov probability * (1 + tfidf_score)','question_id':'Q&A draft identifier','question_type':'Generated question template','question':'Draft question','answer_draft':'Extractive answer requiring human review','evidence_source_ids':'Source identifiers supporting answer','evidence_quote':'Verbatim supporting source quote','notes':'Additional metadata or warnings'}
 write_csv(OUT/'data_dictionary.csv',['column','description','data_type','applicable_record_types'],[{'column':x,'description':defs.get(x,''),'data_type':'float' if x in ('markov_probability','tfidf_score','edge_weight') else 'integer' if x in ('evidence_start','evidence_end') else 'string','applicable_record_types':'EDGE' if x in ('target_concept','target_surface','evidence_start','evidence_end','markov_probability','tfidf_score','edge_weight') else 'QA_DRAFT' if x in ('question_id','question_type','question','answer_draft','evidence_source_ids') else 'ALL'} for x in fields])
 # Leave a *blank* evaluation annotation template. It is not a claimed independently labeled benchmark.
 write_csv(OUT/'INDEPENDENT_EVALUATION_TEMPLATE.csv',['evaluation_id','domain','question','reference_answer','evidence_source_id','reference_url','split','review_status','reviewer','notes'],[])
 from validate_dataset import validate
 report=validate(MAIN)
 report.update({'input_baseline_sources':len(original_sources),'excluded_source_titles':[r['source_title'] for r in excluded_sources], 'excluded_source_rows':len(exclude_ids),'source_risk_flags':dict(Counter(tag for r in src_rows for tag in r['quality_flags'].split(';') if tag)), 'term_risk_flags':dict(Counter(tag for r in term_rows for tag in r['quality_flags'].split(';') if tag)), 'qa_risk_flags':dict(Counter(tag for r in qa_rows for tag in r['quality_flags'].split(';') if tag)),'source_license_status':'PENDING_REVIEW','qa_approval_status':'PENDING_REVIEW','training_ready':False,'independent_benchmark_ready':False})
 (OUT/'dataset_quality_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 meta={'dataset_name':'RAAI-SGC 2.0 Part 1 source-grounded graph dataset','version':'PART1-1.0-candidate','source_file':'input/BASELINE_149_SOURCES.csv','source_sha256':sha(INPUT),'master_filename':MAIN.name,'master_sha256':sha(MAIN),'source_count':len(sources),'edge_count':len(edges),'draft_qa_count':len(qas),'review_status':'UNREVIEWED','license_status':'PENDING_REVIEW','method':'Offline source-row filtering of previously collected revision-tracked Wikipedia records; unchanged source text and graph weights for retained sources','human_approved':False,'benchmark_approved':False,'license_cleared':False}
 (OUT/'dataset_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 (OUT/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.name}' for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {'SHA256SUMS.txt'})+'\n',encoding='utf8')
 print(json.dumps(report,indent=2));return 0 if not report['critical_errors'] else 1

if __name__=='__main__':raise SystemExit(build())
