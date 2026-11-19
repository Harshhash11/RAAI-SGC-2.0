"""Reproducible retrieval-only baseline and weighted graph expansion.

TF-IDF is offline; true semantic embeddings use an explicitly requested
sentence-transformers model. Draft QA rows are never used as labels.
"""
import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

TOKEN = re.compile(r"(?u)\b[a-z][a-z0-9+#.-]*\b", re.I)


def words(s):
    return TOKEN.findall((s or '').lower())


def canonical(s):
    s = re.sub(r'\s*\([^)]*\)', '', s or '')
    return ' '.join(words(s)).strip()


def load_data(path):
    sources = {}
    edges = defaultdict(list)
    qas = 0
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            sid = r['source_id']
            if r['record_type'] == 'SOURCE':
                if sid in sources:
                    raise ValueError('Duplicate SOURCE id: ' + sid)
                sources[sid] = {k:r[k] for k in ['source_id','source_title','source_url','domain','split','source_text','review_status','license_status']}
            elif r['record_type'] == 'EDGE':
                try:
                    w=float(r['edge_weight'])
                except (ValueError, TypeError):
                    raise ValueError('Bad graph edge weight: ' + str(r.get('record_id')))
                if not math.isfinite(w) or w < 0:
                    raise ValueError('Negative/nonfinite graph edge weight')
                edges[sid].append((r['target_concept'], w))
            elif r['record_type'] == 'QA_DRAFT':
                qas += 1
    if not sources:
        raise ValueError('No SOURCE rows found')
    if set(edges)-set(sources):
        raise ValueError('EDGE rows refer to missing SOURCE rows')
    return sources, edges, qas


class HybridRetriever:
    def __init__(self, sources, edges, backend='tfidf', model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.sources=sources
        self.edges=edges
        self.backend=backend
        self.ids=list(sources)
        self.docs=[sources[sid]['source_title']+' . '+sources[sid]['source_title']+' . '+sources[sid]['source_text'] for sid in self.ids]
        if backend == 'tfidf':
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
            except ImportError as e:
                raise RuntimeError('Install: python -m pip install -r requirements.txt') from e
            self.vectorizer=TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, stop_words='english')
            self.matrix=self.vectorizer.fit_transform(self.docs)
        elif backend == 'semantic':
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
            except ImportError as e:
                raise RuntimeError('Install optional semantic requirements: python -m pip install -r requirements-semantic.txt') from e
            self.model=SentenceTransformer(model_name)
            self.matrix=self.model.encode(self.docs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        else:
            raise ValueError('backend must be tfidf or semantic')
        self.titles=defaultdict(list)
        for sid,s in sources.items():
            self.titles[canonical(s['source_title'])].append(sid)
        self.concept_index=defaultdict(set)
        for sid,edge_list in edges.items():
            for term,_ in edge_list:
                for token in set(words(term)):
                    self.concept_index[token].add(sid)

    def similarities(self, question):
        if self.backend=='tfidf':
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(self.vectorizer.transform([question]),self.matrix).ravel()
        import numpy as np
        q=self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        return np.clip(self.matrix @ q,0,1)

    def search(self, question, top_k=5, mode='vector', split='all', alpha=0.25, anchor_k=3):
        if mode not in ('vector','hybrid','unweighted'):
            raise ValueError('mode must be vector, unweighted, or hybrid')
        if not 0<=alpha<=1:
            raise ValueError('alpha must be in [0,1]')
        if top_k<=0: raise ValueError('top_k must be positive')
        if split not in ('all','train','validation','test'):
            raise ValueError('unknown split')
        valid={sid for sid in self.ids if split=='all' or self.sources[sid]['split']==split}
        if not question.strip() or not valid:
            return [], {'graph_activated':False,'graph_source_links':0,'anchors':[]}
        base=self.similarities(question)
        bs={sid:max(0.,float(base[i])) for i,sid in enumerate(self.ids) if sid in valid}
        ranked_anchors=sorted(valid,key=lambda sid:(-bs[sid],sid))
        top_similarity=bs[ranked_anchors[0]] if ranked_anchors else 0.0
        anchors=[sid for sid in ranked_anchors if bs[sid]>0 and bs[sid]>=0.55*top_similarity][:max(0,anchor_k)]
        graph=defaultdict(float)
        why=defaultdict(list)
        # 1-hop graph expansion: anchor SOURCE -> weighted term -> matching SOURCE title.
        # Preserve source eligibility: no cross-split anchor/target when filtering.
        if mode!='vector':
            for aid in anchors:
                strength=bs[aid]
                if strength <= 0: continue
                for term,w in self.edges.get(aid,[]):
                    # Exact graph term to canonical document title; never pretend generic
                    # single-token similarities are verified entity relationships.
                    for target in self.titles.get(canonical(term),[]):
                        if target not in valid or target==aid: continue
                        contribution=strength*(min(w,1.) if mode=='hybrid' else 1.)
                        if contribution>graph[target]:
                            graph[target]=contribution
                            why[target]=[{'from_source':aid,'target_term':term,'edge_weight':round(w,6)}]
        peak=max(graph.values(),default=0.)
        if peak: graph={sid:g/peak for sid,g in graph.items()}
        scored=[]
        for sid in valid:
            g=graph.get(sid,0.)
            similarity=bs[sid]
            # Base score does not shrink unless actual graph paths activate.
            # Bound graph-only promotion by the best query-document similarity.
            # This prevents low-relevance graph targets from outranking
            # a clearly relevant direct match merely through normalization.
            score=similarity + alpha*top_similarity*g if peak else similarity
            if score>0:
                scored.append((score,sid,similarity,g))
        scored.sort(key=lambda x:(-x[0],x[1]))
        results=[]
        for score,sid,sim,g in scored[:top_k]:
            s=self.sources[sid]
            results.append({'score':round(score,6),'vector_score':round(sim,6),
                            'graph_score':round(g,6),'graph_paths':why.get(sid,[]),
                            'source_id':sid,'source_title':s['source_title'],
                            'source_url':s['source_url'],'domain':s['domain'],
                            'split':s['split'],'source_text':s['source_text'],
                            'review_status':s['review_status'],'license_status':s['license_status']})
        return results,{'graph_activated':bool(peak),'graph_source_links':len(graph),
                        'anchors':[self.sources[s]['source_title'] for s in anchors]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/processed/master.csv')
    p.add_argument('--question',required=True)
    p.add_argument('--backend',choices=['tfidf','semantic'],default='tfidf')
    p.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--mode',choices=['vector','unweighted','hybrid'],default='hybrid')
    p.add_argument('--split',choices=['all','train','validation','test'],default='all')
    p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--alpha',type=float,default=.25)
    args=p.parse_args()
    sources,edges,drafts=load_data(args.csv)
    retriever=HybridRetriever(sources,edges,args.backend,args.model)
    begin=time.perf_counter()
    result,diagnostic=retriever.search(args.question,args.top_k,args.mode,args.split,args.alpha)
    out={'backend':args.backend,'mode':args.mode,'question':args.question,
         'source_count':len(sources),'draft_qa_excluded_from_evaluation':drafts,
         'elapsed_retrieval_ms':round((time.perf_counter()-begin)*1000,2),
         'diagnostic':diagnostic,'results':result}
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
