"""Extractive fallback. This is not a generative language model."""
import re
from src.grounding import clean_excerpt, has_evidence, UNKNOWN

def extractive_answer(question, results, max_chars=420):
    if not has_evidence(results):
        return {'answer':UNKNOWN,'citations':[],'mode':'extractive'}
    top=results[0]
    text=clean_excerpt(top.get('source_text',''),max_chars=6000)
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',text)
    toks=set(re.findall(r'[a-z]{4,}',question.lower()))-{'what','which','where','when','does','have','about','explain','difference'}
    ranked=sorted(enumerate(sentences),key=lambda x:(-sum(t in x[1].lower() for t in toks),x[0]))
    answer=(ranked[0][1] if ranked else text)[:max_chars].strip()
    return {'answer':answer or UNKNOWN,'citations':[{'title':top['source_title'],'url':top['source_url'],'source_id':top['source_id']}],'mode':'extractive'}
