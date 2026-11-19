#!/usr/bin/env python3
"""Conservatively export ONLY user-reviewed QA. Refuses to fabricate approvals.
The full master remains the canonical Part 1 artifact.
"""
import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'data'/'review'

def rows(name):
 with (O/name).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def yes(v):return v.strip().upper()=='APPROVED'

def main():
 source_reviews={r['source_id']:r for r in rows('SOURCE_REVIEW_QUEUE.csv')}
 qa_reviews={r['question_id']:r for r in rows('QA_REVIEW_QUEUE.csv')}
 with (ROOT/'data'/'processed'/'master.csv').open(encoding='utf8',newline='') as f:reader=csv.DictReader(f);fieldnames=reader.fieldnames;allrows=list(reader)
 eligible=[]
 for r in allrows:
  if r['record_type']!='QA_DRAFT':continue
  sr=source_reviews.get(r['source_id'],{});q=qa_reviews.get(r['question_id'],{})
  if not all(yes(sr.get(k,'')) for k in ('domain_relevance','factual_accuracy','text_completeness','license_review','attribution_review')):continue
  if not all(yes(q.get(k,'')) for k in ('answer_factuality','answer_relevance','answer_completeness','evidence_adequacy')):continue
  if not (sr.get('reviewer','').strip() and q.get('reviewer','').strip() and q.get('approved_answer','').strip()):continue
  c=r.copy();c['answer_draft']=q['approved_answer'];c['review_status']='APPROVED';c['license_status']='APPROVED';eligible.append(c)
 target=O/'APPROVED_QA_ONLY.csv'
 with target.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(eligible)
 print(f'Approved Q&A records: {len(eligible)}. Output: {target}. No approval inferred from automated checks.')
 return 0

if __name__=='__main__':sys.exit(main())
