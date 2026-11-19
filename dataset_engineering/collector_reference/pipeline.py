"""RAAI-SGC Part 1 V3: independently usable fresh-source dataset pipeline.
CLI: collect, build, validate. No synthetic knowledge is silently substituted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger('raai_v3')
API = 'https://en.wikipedia.org/w/api.php'
SCHEMA_VERSION = '3.4.0'
DOMAINS = {'ai_ml', 'cloud_devops', 'cybersecurity', 'software_engineering'}
FIELDS = [
 'record_id','record_type','dataset_version','domain','secondary_domains','source_id','source_title','source_url',
 'source_revision','retrieved_at','source_provider','license_status','review_status','split','concept_id','concept',
 'source_text','text_hash','target_concept','target_surface','evidence_start','evidence_end','markov_probability','tfidf_score','edge_weight',
 'question_id','question_type','question','answer_draft','evidence_source_ids','evidence_quote','notes'
]
BAD_TITLE = re.compile(r'\((?:novel|film|television series|TV series|album|song|journal|video game|book|band)\)', re.I)
BAD_WORDS = {'system','thing','things','year','years','time','use','way','number','example','people','part','work','various','many','datum','data','field','process','method','software','application','information','resource','type','term','approach','set','form','operation'}


# Human-maintained normalization aliases. These are candidate labels, not factual claims.
GLOSSARY_FILE = ROOT/'config/technical_glossary.json'

@lru_cache(maxsize=1)
def glossary():
    try:
        return load_json(GLOSSARY_FILE)
    except FileNotFoundError:
        return {'canonical_aliases': {}, 'exclude_exact': []}


def canonical_concept(lemma_phrase, surface_phrase=''):
    """Canonicalize common spaCy lemma artifacts without changing source evidence."""
    config = glossary()
    aliases = config.get('canonical_aliases', {})
    raw = clean_phrase(lemma_phrase)
    surface = clean_phrase(surface_phrase)
    if not raw or raw in set(config.get('exclude_exact', [])):
        return ''
    # Technical 'data' is not the biological singular 'datum'.
    raw = re.sub(r'\bdatum\b', 'data', raw)
    result = aliases.get(raw, aliases.get(surface, raw))
    if result in set(config.get('exclude_exact', [])):
        return ''
    return clean_phrase(result)


def expansion_candidate(title, domain, seed_names):
    """Conservative ranking for real Wikipedia outlinks; never invent page names."""
    title = normal_text(title)
    if not title or len(title)>85 or BAD_TITLE.search(title) or ':' in title:
        return False
    if title.casefold() in seed_names:
        return True
    tokens = set(re.findall(r'[a-z]{3,}',title.casefold()))
    keywords = {
      'ai_ml': {'learning','neural','artificial','intelligence','machine','classification','regression','transformer','model','embedding','gradient','clustering','language','algorithm','optimization','retrieval','inference'},
      'cloud_devops': {'cloud','container','kubernetes','docker','devops','deployment','integration','orchestration','microservice','computing','virtualization','distributed','serverless','infrastructure','load'},
      'cybersecurity': {'security','cryptography','encryption','authentication','authorization','vulnerability','malware','attack','firewall','network','phishing','identity','access','tls','threat'},
      'software_engineering': {'software','programming','testing','database','architecture','api','version','control','integration','development','design','git','compiler','code','repository'}
    }
    return bool(tokens & keywords[domain])



def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def hash_text(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()


def normal_text(s): return re.sub(r'\s+',' ', str(s or '')).strip()


def utc_now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')


def collect(args):
    if not args.contact or ('@' not in args.contact and 'http' not in args.contact):
        raise ValueError('Use --contact with a real email or project URL for Wikimedia User-Agent.')
    import requests
    client = requests.Session()
    client.headers.update({'User-Agent': f'RAAI-SGC-DatasetResearch/3.4 ({args.contact}) python-requests', 'Accept':'application/json'})
    seeds = load_json(ROOT/'config/seeds.json')
    # Round-robin avoids filling the page budget disproportionately from one domain.
    domains_in_order = ['ai_ml','cloud_devops','cybersecurity','software_engineering']
    max_seed_count = max(len(seeds[d]) for d in domains_in_order)
    q = deque((seeds[d][j], d, 0) for j in range(max_seed_count) for d in domains_in_order if j<len(seeds[d]))
    seed_names={v.casefold() for values in seeds.values() for v in values}
    visited=set(); accepted=[]; seen_ids=set(); errors=[]
    cache_dir = ROOT/'data/raw/cache'; cache_dir.mkdir(parents=True,exist_ok=True)
    while q and len(accepted)<args.max_pages:
        title,domain,depth=q.popleft()
        key = title.casefold()
        if key in visited: continue
        visited.add(key)
        cache_path = cache_dir/(hash_text(key)[:24]+'.json')
        try:
            if cache_path.exists() and not args.refresh_cache: raw=load_json(cache_path)
            else:
                params={'action':'query','format':'json','formatversion':2,'redirects':1,'prop':'extracts|info|revisions|categories|links',
                        'inprop':'url','rvprop':'ids|timestamp','rvlimit':1,'cllimit':30,'pllimit':100,
                        'explaintext':1,'exintro':1,'titles':title}
                for attempt in range(4):
                    try:
                        resp=client.get(API,params=params,timeout=25)
                        if resp.status_code in (429,500,502,503,504):
                            wait=float(resp.headers.get('Retry-After') or min(30,2**attempt))
                            LOG.warning('HTTP %s for %s; retrying in %.1fs',resp.status_code,title,wait)
                            time.sleep(min(wait,60)); continue
                        resp.raise_for_status(); raw=resp.json(); break
                    except (requests.RequestException,ValueError):
                        if attempt==3: raise
                        time.sleep(min(2**attempt,12))
                else: raise RuntimeError(f'API did not recover for {title}')
                write_json(cache_path,raw); time.sleep(args.delay)
            pages=raw.get('query',{}).get('pages',[])
            if not pages: continue
            p=pages[0]; pid=p.get('pageid')
            body=normal_text(p.get('extract',''))
            name=normal_text(p.get('title',title))
            if not pid or len(body)<85 or BAD_TITLE.search(name) or p.get('missing'): continue
            if pid in seen_ids: continue
            cats=[c.get('title','').replace('Category:','') for c in p.get('categories',[])]
            if any(re.search(r'^(?:\d{4} .*films|.* television series|.* albums)$',c,re.I) for c in cats): continue
            rev=p.get('revisions',[{}])[0]
            source={'source_id':f'wiki:{pid}','domain':domain,'source_title':name,'source_url':p.get('fullurl') or 'https://en.wikipedia.org/wiki/'+quote(name.replace(' ','_')),
             'revision_id':str(rev.get('revid','')),'revision_timestamp':rev.get('timestamp',''), 'retrieved_at':utc_now(),
             'provider':'English Wikipedia','license_status':'REVIEW_CC_BY_SA','text':body,'categories':cats,'depth':depth}
            accepted.append(source); seen_ids.add(pid)
            LOG.info('%d/%d [%s] %s',len(accepted),args.max_pages,domain,name)
            # Expansion candidates are titles linked from the actual article, not guessed names.
            # Filters are heuristics; relevance is not claimed without review.
            if depth < args.depth:
                links = [x.get('title','') for x in p.get('links',[]) if x.get('ns',0)==0]
                links = sorted(set(links), key=lambda t:(t.casefold() not in seed_names,t.casefold()))
                kept=0
                for label in links:
                    if expansion_candidate(label,domain,seed_names) and label.casefold()!=name.casefold():
                        q.append((label,domain,depth+1));kept+=1
                        if kept>=args.branch:break
        except Exception as ex:
            errors.append({'title':title,'error':str(ex)[:500]})
            LOG.warning('Skipped %s: %s',title,ex)
    out=ROOT/'data/raw/sources.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for x in accepted: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    write_json(ROOT/'data/raw/collection_report.json',{'collected':len(accepted),'attempted_titles':len(visited),'errors':errors,
      'warning':'Collection is unreviewed. Check article relevance and license before training.'})
    if not accepted: raise RuntimeError('No live sources collected; check connection and API errors. No fabricated fallback.')
    print(f'Collected {len(accepted)} source records → {out}')


def clean_phrase(phrase):
    """Conservatively normalize term candidates; preserve acronyms and meaningful phrases."""
    phrase = normal_text(phrase).lower()
    if re.search(r'[(){}\[\]\\]|(?<!\w)[^\w\s-]|\d', phrase):
        return ''
    phrase = re.sub(r'\s*-\s*', '-', phrase).strip(' -')
    if not (4 <= len(phrase) <= 70) or len(phrase.split()) > 5:
        return ''
    if phrase in BAD_WORDS or all(t in BAD_WORDS for t in phrase.split()):
        return ''
    return phrase


def extract_mentions(text, nlp):
    """Extract noun phrase occurrences with exact supporting source sentences.

    Each returned sentence is taken verbatim from the text being processed;
    morphological lemmatization changes the term key, not the evidence.
    """
    doc = nlp(text[:15000])
    result = []
    occupied = set()
    spans = sorted(list(doc.noun_chunks) + list(doc.ents), key=lambda sp: (sp.start, -(sp.end-sp.start)))
    for span in spans:
        if any(j in occupied for j in range(span.start, span.end)):
            continue
        toks = [t for t in span if not t.is_punct and not t.is_space]
        if not toks or len(toks) > 5 or any(t.text in ('(', ')', '[', ']', '{', '}') for t in toks):
            continue
        core = [t for t in toks if not t.is_stop]
        if not core or not any(t.pos_ in ('NOUN', 'PROPN') for t in core):
            continue
        while toks and toks[0].is_stop:
            toks.pop(0)
        while toks and toks[-1].is_stop:
            toks.pop()
        if not toks:
            continue
        phrase = ' '.join(t.lemma_.lower() if t.pos_ != 'PROPN' else t.text.lower() for t in toks)
        phrase = canonical_concept(phrase, ' '.join(t.text for t in toks))
        if not phrase:
            continue
        sentence = normal_text(span.sent.text)
        if not sentence or sentence not in normal_text(text):
            continue
        # Store the original sentence span, source-relative offsets, and surface form.
        # These allow a reviewer to trace lemmatized terms to the exact original passage.
        start, end = span.sent.start_char, span.sent.end_char
        surface = normal_text(text[toks[0].idx:toks[-1].idx + len(toks[-1].text)])
        result.append((phrase, sentence, surface, start, end))
        occupied.update(range(span.start, span.end))
    return result[:250]


def extract_terms(text, nlp):
    return [item[0] for item in extract_mentions(text, nlp)]


def source_is_truncated(text):
    return bool(re.search(r'(?:\.\.\.|…)[\s]*$', normal_text(text)))


def balanced_splits(docs):
    """Deterministic domain-stratified source grouping; 8/1/1 for each 10-source pilot."""
    by_domain=defaultdict(list)
    for d in docs: by_domain[d['domain']].append(d['source_id'])
    result={}
    for domain,ids in by_domain.items():
        ids=sorted(ids,key=lambda sid:hash_text('v3.1:'+sid))
        n=len(ids)
        if n>=10:
            nv=max(1,round(n*0.10)); nt=max(1,round(n*0.10))
        elif n>=3: nv=1;nt=1
        else: nv=0;nt=0
        train=n-nv-nt
        for i,sid in enumerate(ids):
            result[sid]='train' if i<train else ('validation' if i<train+nv else 'test')
    return result

def deterministic_split(group):
    x=int(hash_text(group)[:8],16)%100
    return 'train' if x<80 else ('validation' if x<90 else 'test')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except ValueError as exc:raise ValueError(f'Invalid JSONL at line {i}') from exc


def build(args):
    try:
        import spacy
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as ex: raise RuntimeError('Install dependencies: python3 -m pip install -r requirements.txt') from ex
    try:nlp=spacy.load('en_core_web_sm')
    except OSError as ex:raise RuntimeError('Install spaCy model: python3 -m spacy download en_core_web_sm') from ex
    path=Path(args.sources) if args.sources else ROOT/'data/raw/sources.jsonl'
    if not path.exists(): raise FileNotFoundError(f'No raw sources at {path}; run collect first.')
    raw=list(read_jsonl(path)); dedup={}; rejected=[]
    for d in raw:
        title=normal_text(d.get('source_title','')); body=normal_text(d.get('text',''))
        if d.get('domain') not in DOMAINS or not d.get('source_url','').startswith('https://') or len(body)<85 or BAD_TITLE.search(title):
            rejected.append({'source_id':d.get('source_id'),'reason':'invalid_domain/url/text_or_disambiguation'});continue
        if not d.get('source_id') or not d.get('revision_id'):
            rejected.append({'source_id':d.get('source_id'),'reason':'missing_source_id_or_revision'});continue
        source_key=d['source_id']
        if source_key not in dedup: dedup[source_key]=d
    docs=list(dedup.values()); docs.sort(key=lambda d:d['source_id'])
    if not docs: raise ValueError('No valid source records. Cannot build dataset.')
    mentions={d['source_id']:extract_mentions(normal_text(d['text']),nlp) for d in docs}
    phrases={sid:[m[0] for m in pairs] for sid,pairs in mentions.items()}
    # Each source contributes a document of detected technical terms, preserving phrase boundaries via token lists.
    corpus=[' '.join(t.replace(' ','_') for t in phrases[d['source_id']]) or '__no_terms__' for d in docs]
    tfidf=TfidfVectorizer(token_pattern=r'(?u)\b[^\s]+\b',lowercase=False,norm='l2')
    matrix=tfidf.fit_transform(corpus)
    vocab=tfidf.vocabulary_
    out_rows=[]; source_reviews=[]; qa_reviews=[]; edge_count=0; qa_count=0; truncated_sources=[]; missing_evidence=[]
    splits=balanced_splits(docs)
    for i,d in enumerate(docs):
        sid=d['source_id']; split=splits[sid]
        terms=phrases[sid]; counts=Counter(terms)
        evidence_by_term={}
        for term, sentence, surface, start, end in mentions[sid]:
            evidence_by_term.setdefault(term, (sentence, surface, start, end))
        truncated=source_is_truncated(d['text'])
        if truncated: truncated_sources.append(sid)
        # Exclude source-title self-loops from edge weights; normalize remaining terms.
        source_name=clean_phrase(d['source_title'])
        counts={term:freq for term,freq in counts.items() if term!=source_name}
        denominator=sum(counts.values())
        cid='concept:'+hash_text(d['source_title'].casefold())[:16]
        base=dict(record_id='',record_type='',dataset_version=SCHEMA_VERSION,domain=d['domain'],secondary_domains='',
          source_id=sid,source_title=d['source_title'],source_url=d['source_url'],source_revision=d['revision_id'],
          retrieved_at=d.get('retrieved_at',''),source_provider=d.get('provider','English Wikipedia'),license_status=d.get('license_status','REVIEW_CC_BY_SA'),
          review_status='UNREVIEWED',split=split,concept_id=cid,concept=d['source_title'],source_text=normal_text(d['text']),
          text_hash=hash_text(normal_text(d['text'])),target_concept='',target_surface='',evidence_start='',evidence_end='',markov_probability='',tfidf_score='',edge_weight='',
          question_id='',question_type='',question='',answer_draft='',evidence_source_ids=sid,evidence_quote='',notes='')
        def emit(t,**updates):
            row=base.copy();row.update(updates);row['record_type']=t
            row['record_id']=f'{t.lower()}:{hash_text(sid+str(len(out_rows))+t)[:16]}'
            out_rows.append(row)
        emit('SOURCE',notes=('TRUNCATED_SOURCE; ' if truncated else '')+'Article text is original source; not human-approved')
        source_reviews.append({'source_id':sid,'source_title':d['source_title'],'domain':d['domain'],'source_url':d['source_url'],
          'source_completeness':'TRUNCATED' if truncated else 'CHECK','technical_relevance':'PENDING','article_disambiguation':'PENDING','license_permission':'PENDING','source_accuracy':'PENDING','reviewer':'','review_notes':''})
        for term,freq in sorted(counts.items()):
            if not denominator:continue
            p=freq/denominator; token=term.replace(' ','_'); t=float(matrix[i,vocab[token]]) if token in vocab else 0.0
            weight=p*(1+t)
            quote,surface,start,end=evidence_by_term[term]
            emit('EDGE',target_concept=term,target_surface=surface,evidence_start=str(start),evidence_end=str(end),
               markov_probability=f'{p:.12g}',tfidf_score=f'{t:.12g}',edge_weight=f'{weight:.12g}',
               evidence_quote=quote,notes='Exact source span and original phrase; association only, not verified factual relation')
            if term not in evidence_by_term: missing_evidence.append((sid,term))
            edge_count+=1
        # Extractive draft: no hallucinated model-written answers, not a gold QA item.
        answer=normal_text(d['text'])
        # Short extractive draft, not the full Wikipedia introduction.
        sentences = list(nlp(answer).sents) if callable(nlp) else re.split(r'(?<=[.!?])\s+(?=[A-Z])',answer)
        answer=normal_text(' '.join(str(z) for z in sentences[:2]))
        if answer.endswith('...'): answer=answer[:-3].rstrip()
        question=f'What is {d["source_title"]}?' 
        # The full answer is its own evidence; never truncate an evidence quotation.
        emit('QA_DRAFT',question_id='qa:'+hash_text(sid+'definition')[:16],question_type='definition',question=question,
             answer_draft=answer,evidence_quote=answer,notes='Draft only, verbatim source extract; requires review')
        qa_count+=1
        # A second, explicitly extractive draft based on one edge's actual source sentence.
        for term in sorted(counts, key=lambda t: (-counts[t], t)):
            evidence = evidence_by_term.get(term)
            quote = evidence[0] if evidence else ''
            if quote and len(quote) >= 35 and not quote.endswith('...'):
                qid='qa:'+hash_text(sid+'mention:'+term)[:16]
                q=f'What does the source explain about {term} in the context of {d["source_title"]}?'
                emit('QA_DRAFT',question_id=qid,question_type='technical_context',question=q,
                     answer_draft=quote,evidence_quote=quote,
                     notes='Exact extractive evidence sentence, not an independently verified explanation')
                qa_count+=1
                qa_reviews.append({'question_id':qid,'source_id':sid,'question':q,
                    'answer_draft':quote,'grounded_in_source':'PENDING','factual_correctness':'PENDING',
                    'answer_completeness':'PENDING','reviewer':'','review_notes':''})
                break
        qa_reviews.append({'question_id':'qa:'+hash_text(sid+'definition')[:16],'source_id':sid,'question':question,
          'answer_draft':answer,'grounded_in_source':'PENDING','factual_correctness':'PENDING',
          'answer_completeness':'PENDING','reviewer':'','review_notes':''})
    output=Path(args.output) if args.output else ROOT/'output/RAAI_SGC_2_MASTER_DATASET_V3_4.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(out_rows)
    review_path=output.parent/'SOURCE_REVIEW_QUEUE.csv'
    with review_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(source_reviews[0]));w.writeheader();w.writerows(source_reviews)
    with (output.parent/'QA_REVIEW_QUEUE.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(qa_reviews[0]));w.writeheader();w.writerows(qa_reviews)
    dictionary=[{'column':field,'description':{'record_type':'SOURCE, EDGE, or QA_DRAFT','source_id':'Stable Wikipedia page ID','split':'Deterministic source-group split','review_status':'UNREVIEWED until human adjudication','edge_weight':'Markov probability * (1 + TF-IDF) for EDGE rows'}.get(field,'See DATASET_CONTRACT.md'),
                 'nullable':'yes' if field in ('secondary_domains','target_concept','markov_probability','tfidf_score','edge_weight','question_id','question_type','question','answer_draft','notes') else 'record-type dependent'} for field in FIELDS]
    with (output.parent/'data_dictionary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['column','description','nullable']);w.writeheader();w.writerows(dictionary)
    metadata={'schema_version':SCHEMA_VERSION,'generated_at':utc_now(),'source_count':len(docs),'edge_count':edge_count,'draft_qa_count':qa_count,
       'master_row_count':len(out_rows),'excluded_records':rejected,'truncated_source_ids':truncated_sources,'missing_edge_evidence':missing_evidence,'source_status':'UNREVIEWED','qa_status':'DRAFT_NOT_GOLD',
       'data_license_status':'REVIEW_CC_BY_SA','split_policy':'Deterministic domain-stratified source-group split; all records for a source stay in one split',
       'input_path':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'master_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'important_limitations':[
       'Original study Markov/TF-IDF methods are approximated for phrase-preserving extraction; verify correspondence before claiming reproduction.',
       'Source-based QA drafts are not independent gold test questions.',
       'Original technical phrase and exact sentence offsets are stored to support auditing of lemmatized term mappings.',
       'Only definition and technical-context extractive questions are generated; no unsupported comparison or troubleshooting claims are invented.',
       'Source license and article relevance need human review.',
       'Prior V3 raw extracts may already be truncated; recollect with --refresh-cache to repair. Wikipedia introductions can legitimately end with ellipses, so review flags manually.',
       'Candidate extracted terms and weighted associations remain unverified, not factual knowledge graph triples.']}
    # Auditable term inventory; all candidates are unreviewed by default.
    term_index=defaultdict(lambda:{'source_ids':set(),'occurrences':0,'domains':set()})
    for d in docs:
        for term in phrases[d['source_id']]:
            x=term_index[term]; x['source_ids'].add(d['source_id']);x['occurrences']+=1;x['domains'].add(d['domain'])
    term_review=[{'target_concept':term,'source_count':len(item['source_ids']),
      'occurrences':item['occurrences'],'domains':'|'.join(sorted(item['domains'])),
      'quality_decision':'PENDING','canonical_override':'','reviewer':'','review_notes':''}
      for term,item in sorted(term_index.items())]
    with (output.parent/'TERM_REVIEW_QUEUE.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['target_concept','source_count','occurrences','domains','quality_decision','canonical_override','reviewer','review_notes']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(term_review)
    metadata['term_review_count']=len(term_review)
    metadata['glossary_sha256']=hashlib.sha256(GLOSSARY_FILE.read_bytes()).hexdigest() if GLOSSARY_FILE.exists() else None
    metadata['recollection_scope']='Wikipedia article introductions only; expansion via actual outbound article links'
    write_json(output.parent/'dataset_metadata.json',metadata)
    print(f'Built {len(out_rows)} rows from {len(docs)} sources ({edge_count} edges, {qa_count} draft Q&A) → {output}')
    validate_file(output,output.parent/'dataset_quality_report.json')


def validate_file(csv_path,report_path=None):
    path=Path(csv_path)
    if not path.exists():raise FileNotFoundError(path)
    errors=[]; warnings=[]; counts=Counter(); domains=Counter(); ids=set(); src_splits=defaultdict(set); groups=defaultdict(set); n=0
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS: errors.append('Unexpected CSV schema or column order')
        for line,r in enumerate(reader,2):
            n+=1
            if r.get('record_id') in ids:errors.append(f'Duplicate record_id at line {line}')
            ids.add(r.get('record_id'))
            if not r.get('source_id') or not r.get('source_url','').startswith('https://'):errors.append(f'Invalid source identity at line {line}')
            if r.get('domain') not in DOMAINS:errors.append(f'Invalid domain at line {line}')
            if r.get('split') not in ('train','validation','test'):errors.append(f'Invalid split at line {line}')
            counts[r.get('record_type')]+=1;domains[r.get('domain')]+=1
            src_splits[r.get('source_id')].add(r.get('split'));groups[r.get('source_id')].add(r.get('record_type'))
            if r.get('record_type')=='EDGE':
                if not r.get('evidence_quote') or r['evidence_quote'] not in r.get('source_text',''):
                    errors.append(f'EDGE evidence missing from source at line {line}')
                try:
                    start,end=int(r['evidence_start']),int(r['evidence_end'])
                    if (start<0 or end<=start or r['source_text'][start:end]!=r['evidence_quote']):
                        errors.append(f'EDGE offset invalid at line {line}')
                    if not r['target_surface'] or r['target_surface'].casefold() not in r['evidence_quote'].casefold():
                        errors.append(f'EDGE original phrase not in evidence at line {line}')
                except (ValueError, KeyError): errors.append(f'EDGE missing or nonnumeric offsets at line {line}')
                if r.get('concept','').casefold()==r.get('target_concept','').casefold():
                    errors.append(f'Self-loop at line {line}')
                try:
                    p,t,w=map(float,(r['markov_probability'],r['tfidf_score'],r['edge_weight']))
                    if not (0<=p<=1 and 0<=t<=1 and math.isclose(w,p*(1+t),rel_tol=1e-8,abs_tol=1e-9)):
                        errors.append(f'Edge weight invalid at line {line}')
                except (ValueError,KeyError):errors.append(f'Invalid edge numeric values at line {line}')
            if r.get('text_hash') and hash_text(r.get('source_text','')) != r.get('text_hash'):
                errors.append(f'Source hash mismatch at line {line}')
            if r.get('record_type')=='QA_DRAFT' and (not r.get('question') or not r.get('answer_draft')):
                errors.append(f'Incomplete Q&A at line {line}')
            if r.get('record_type')=='QA_DRAFT':
                if not r.get('evidence_quote') or r['evidence_quote'] not in r.get('source_text',''):
                    errors.append(f'Q&A evidence missing from source at line {line}')
                if r.get('answer_draft') not in r.get('evidence_quote',''):
                    errors.append(f'Q&A answer not fully supported by stored evidence at line {line}')
            if r.get('review_status')!='APPROVED' and r.get('record_type')=='QA_DRAFT':
                pass
    for sid,sp in src_splits.items():
        if len(sp)!=1:errors.append(f'Source split leakage: {sid}')
    for sid,typ in groups.items():
        if 'SOURCE' not in typ:errors.append(f'Orphan records without SOURCE: {sid}')
    if counts.get('QA_DRAFT',0):warnings.append('Draft Q&A not independently verified; not approved for training or gold testing.')
    if len(src_splits)<100:warnings.append('Pilot corpus has fewer than 100 sources; evaluation split may be too small.')
    if not counts.get('SOURCE'):errors.append('No SOURCE records')
    report={'status':'FAIL' if errors else 'CONDITIONAL_PASS','schema_version':SCHEMA_VERSION,
      'file':str(path),'rows':n,'unique_sources':len(src_splits),'record_types':dict(counts),'domains_by_row':dict(domains),
      'errors':errors[:100],'warnings':warnings,'review_approval':'NOT_VERIFIED',
      'evaluation_readiness':'NOT_READY','training_readiness':'NOT_READY'}
    if report_path:write_json(report_path,report)
    print(json.dumps(report,indent=2))
    if errors:raise ValueError(f'Validation failed: {len(errors)} errors')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log-level',default='INFO')
    sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('collect',help='Fetch real Wikipedia source records (requires internet)')
    c.add_argument('--contact',required=True);c.add_argument('--max-pages',type=int,default=80)
    c.add_argument('--depth',type=int,choices=[0,1,2],default=0);c.add_argument('--branch',type=int,default=3)
    c.add_argument('--delay',type=float,default=1.5)
    c.add_argument('--refresh-cache',action='store_true',help='Force new Wikimedia API fetches instead of reusing cached pages')
    b=sub.add_parser('build',help='Generate one typed master CSV from collected raw sources')
    b.add_argument('--sources',default=None);b.add_argument('--output',default=None)
    v=sub.add_parser('validate',help='Validate master CSV')
    v.add_argument('--csv',default=str(ROOT/'output/RAAI_SGC_2_MASTER_DATASET_V3_4.csv'))
    a=ap.parse_args();logging.basicConfig(level=getattr(logging,a.log_level.upper(),logging.INFO),format='%(levelname)s %(message)s')
    if a.cmd=='collect':collect(a)
    elif a.cmd=='build':build(a)
    else:validate_file(a.csv,Path(a.csv).parent/'dataset_quality_report.json')

if __name__=='__main__':main()
