#!/usr/bin/env python3
"""Generate *unapproved* Q&A revision suggestions from local 4-bit Qwen.

No input review decisions are modified. Never promotes model guesses to gold labels.
Run --dry-run first; use .venv-llm for GPU generation.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data/review/QA_AUTO_REVIEW.csv'
FALLBACK_INPUT = ROOT / 'data/review/QA_REVIEW_QUEUE.csv'
DEFAULT_OUTPUT = ROOT / 'data/review/QA_AI_SUGGESTIONS.csv'
MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
OUT_COLS = ['question_id', 'source_id', 'source_title', 'domain', 'split', 'question_type',
            'original_question', 'original_answer', 'original_evidence', 'original_quality_flags',
            'auto_review_flags', 'suggested_question', 'suggested_answer',
            'ai_review_notes', 'generation_status', 'model_id', 'human_review_status']
SYSTEM = ("You are an IT Q&A editor. Only use the provided SOURCE EVIDENCE. "
          "Do not invent unsupported details. Output exactly three plain-text lines, "
          "without JSON or markdown, using these labels: "
          "QUESTION: <one natural specific technical question>\n"
          "ANSWER: <at most two concise sentences supported by source evidence>\n"
          "NOTES: <uncertainties or limitations, or 'Requires human verification'>. "
          "If evidence is insufficient, write ANSWER: INSUFFICIENT_EVIDENCE. "
          "Do not repeat the original generic question template. "
          "All suggestions require human review.")


def flags(row):
    existing = row.get('auto_review_flags') or row.get('quality_flags') or ''
    found = {s.strip() for s in re.split(r'[;,]', existing) if s.strip()}
    q = (row.get('question') or '').lower()
    if 'what does the source explain' in q:
        found.add('TEMPLATE_CONTEXT_QUESTION')
    if not (row.get('evidence_quote') or '').strip():
        found.add('MISSING_EVIDENCE')
    if not (row.get('answer_draft') or '').strip():
        found.add('MISSING_ANSWER')
    return sorted(found)


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    needed = {'question_id','question','answer_draft','evidence_quote','split','source_id'}
    if not rows:
        raise ValueError(f'No data rows found: {path}')
    missing = needed - rows[0].keys()
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    return rows


def base_record(row, model):
    return {
        'question_id': row['question_id'], 'source_id': row.get('source_id',''),
        'source_title': row.get('source_title',''), 'domain': row.get('domain',''),
        'split': row.get('split',''), 'question_type': row.get('question_type',''),
        'original_question': row.get('question',''),
        'original_answer': row.get('answer_draft',''),
        'original_evidence': row.get('evidence_quote',''),
        'original_quality_flags': row.get('quality_flags',''),
        'auto_review_flags': ';'.join(flags(row)),
        'suggested_question': '', 'suggested_answer': '', 'ai_review_notes': '',
        'generation_status': 'NOT_RUN', 'model_id': model,
        'human_review_status': 'PENDING',
    }


def choose(rows, split, only_flagged, limit):
    filtered = [r for r in rows if (split == 'all' or r['split'] == split)
                and (not only_flagged or flags(r))]
    # Flagged technical-context records come first, then stable original order.
    filtered.sort(key=lambda r: ('TEMPLATE_CONTEXT_QUESTION' not in flags(r),
                                 r.get('question_id','')))
    return filtered[:limit] if limit else filtered


def _normalize_fields(item):
    """Accept common Qwen JSON key variations, but never invent a missing answer."""
    if not isinstance(item, dict):
        raise ValueError('Model response is not a JSON object')
    q = item.get('suggested_question', item.get('question'))
    a = item.get('suggested_answer', item.get('answer'))
    n = item.get('ai_review_notes', item.get('notes', item.get('reason',
        'AI proposal; check against source before approval.')))
    if not isinstance(q, str) or not isinstance(a, str):
        raise ValueError('Model response missing a recognizable question/answer pair')
    if not isinstance(n, str):
        n = 'AI proposal; check against source before approval.'
    return {'suggested_question': q.strip(), 'suggested_answer': a.strip(),
            'ai_review_notes': n.strip()}


def parse_json(text):
    """Parse strictly valid JSON, allowing old and new field names."""
    raw = text.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.I)
        raw = re.sub(r'\s*```$', '', raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find('{'), raw.rfind('}')
        if start < 0 or end <= start:
            raise ValueError('No JSON object in model response')
        try:
            obj = json.loads(raw[start:end+1])
        except json.JSONDecodeError as exc:
            raise ValueError('Invalid JSON object in model response') from exc
    return _normalize_fields(obj)


def parse_suggestion(text):
    """Prefer three labeled lines; fall back to valid JSON produced by older prompts.

    An arbitrary key/value pair is deliberately rejected: it often just repeats
    the original template rather than supplying a supported QA pair.
    """
    text = text.strip()
    labels = re.search(r'(?im)^\s*QUESTION\s*:', text)
    if labels:
        pieces = re.split(r'(?im)^\s*(QUESTION|ANSWER|NOTES)\s*:\s*', text)
        fields = {}
        for i in range(1, len(pieces) - 1, 2):
            fields[pieces[i].upper()] = pieces[i+1].strip()
        q, a = fields.get('QUESTION', ''), fields.get('ANSWER', '')
        if not q or not a:
            raise ValueError('Missing QUESTION or ANSWER labeled section')
        return _normalize_fields({'question': q, 'answer': a,
                                  'notes': fields.get('NOTES', 'Requires human verification.')})
    return parse_json(text)


def suggestion_status(suggestion):
    q, a = suggestion['suggested_question'], suggestion['suggested_answer']
    if not q or not a or a.upper().startswith('INSUFFICIENT_EVIDENCE'):
        return 'NEEDS_MANUAL_REVIEW'
    if 'what does the source explain' in q.lower():
        return 'NEEDS_MANUAL_REVIEW'
    if not q.rstrip().endswith('?'):
        return 'NEEDS_MANUAL_REVIEW'
    return 'SUGGESTED'


def user_prompt(row):
    return (f"Source title: {row.get('source_title','')}\n"
            f"Domain: {row.get('domain','')}\n"
            f"Question category: {row.get('question_type','')}\n"
            f"Original question: {row.get('question','')}\n"
            f"Draft answer: {row.get('answer_draft','')}\n"
            f"SOURCE EVIDENCE (exclusive authority):\n{row.get('evidence_quote','')}\n\n"
            "Rewrite the question to be meaningful and specific. Propose a concise answer "
            "fully supported by the evidence; if the evidence is insufficient, write INSUFFICIENT_EVIDENCE. "
            "Use the three labeled lines QUESTION:, ANSWER:, NOTES:. Do not output JSON or Markdown.")


def load_model(model_id):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this 4-bit generation mode; use --dry-run on CPU')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={'':0})
    model.eval()
    return tokenizer, model


def generate(row, tokenizer, model, max_input_tokens, max_new_tokens):
    import torch
    messages = [{'role':'system','content':SYSTEM},
                {'role':'user','content':user_prompt(row)}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True,
                                            return_dict=True,return_tensors='pt',
                                            truncation=True,max_length=max_input_tokens)
    encoded = {k:v.to('cuda') for k,v in encoded.items()}
    with torch.inference_mode():
        output = model.generate(**encoded,max_new_tokens=max_new_tokens,
                                do_sample=False,pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0][encoded['input_ids'].shape[-1]:],skip_special_tokens=True)
    try:
        return parse_suggestion(text)
    except ValueError as exc:
        # Preserve a short raw excerpt for diagnosis; do not treat malformed JSON as approved.
        raise ValueError(f'{exc}; model_output_excerpt={text[:400]!r}') from exc


def write_csv_atomic(out, records):
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=OUT_COLS,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,out)


def prior_records(output):
    if not output.is_file():return []
    with open(output,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def run(args):
    path = args.input or (DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT)
    rows = read_rows(path)
    selected = choose(rows,args.split,not args.all_questions,args.limit)
    if args.dry_run:
        count = Counter(';'.join(flags(r)) or 'NONE' for r in selected)
        print('DRY RUN - no model loaded, no files written')
        print('Input:',path,'Total rows:',len(rows),'Selected:',len(selected))
        print('Split:',args.split,'Only flagged:',not args.all_questions)
        print('Flag groups:',dict(count))
        return 0
    if not selected:
        print('No rows selected; nothing to generate.')
        return 0
    # Keep prior progress even when a batch is interrupted; do not rewrite other decisions.
    old = prior_records(args.output)
    if old and not args.resume and not args.overwrite:
        raise RuntimeError(f'{args.output} already exists. Use --resume or --overwrite.')
    previous = {r['question_id']:r for r in old} if args.resume else {}
    if args.resume:
        mismatch = [r for r in selected if r['question_id'] in previous and
                    (r.get('question','') != previous[r['question_id']].get('original_question','') or
                     r.get('evidence_quote','') != previous[r['question_id']].get('original_evidence',''))]
        if mismatch:
            raise RuntimeError('Existing suggestion input changed. Back up output and use --overwrite after review.')
    # On --resume, retry failed generations rather than permanently skipping them.
    failed_statuses = {'GENERATION_ERROR', 'NOT_RUN'}
    todo = [r for r in selected if r['question_id'] not in previous
            or previous[r['question_id']].get('generation_status') in failed_statuses]
    if not todo:
        print('All selected records already in output; nothing to generate.')
        return 0
    print(f'Loading 4-bit model for {len(todo)} records: {args.model}')
    tokenizer, model = load_model(args.model)
    records = old[:] if args.resume else []
    indexes = {r['question_id']: i for i, r in enumerate(records)}
    for index,row in enumerate(todo,1):
        result = base_record(row,args.model)
        if not row.get('evidence_quote','').strip():
            result['generation_status']='SKIPPED_MISSING_EVIDENCE'
            result['ai_review_notes']='No source evidence; manual research required.'
        else:
            try:
                suggestion = generate(row,tokenizer,model,args.max_input_tokens,args.max_new_tokens)
                result.update(suggestion)
                result['generation_status'] = suggestion_status(suggestion)
            except Exception as exc:
                result['generation_status']='GENERATION_ERROR'
                result['ai_review_notes']=str(exc)[:800]
        if row['question_id'] in indexes:
            # Do not silently overwrite a human review decision.
            prior = records[indexes[row['question_id']]]
            if prior.get('human_review_status', 'PENDING') != 'PENDING':
                raise RuntimeError('Refusing to replace a human-reviewed record')
            records[indexes[row['question_id']]] = result
        else:
            indexes[row['question_id']] = len(records)
            records.append(result)
        write_csv_atomic(args.output,records)
        print(f'[{index}/{len(todo)}] {row["question_id"]}: {result["generation_status"]}',flush=True)
        if result['generation_status'] == 'GENERATION_ERROR':
            print('    Error:', result['ai_review_notes'][:300],flush=True)
    print('Saved:',args.output)
    print('All suggestions remain human_review_status=PENDING; no approval exports modified.')
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=None)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--model',default=MODEL_ID)
    p.add_argument('--split',choices=('train','validation','test','all'),default='train')
    p.add_argument('--all-questions',action='store_true',help='Include unflagged questions too')
    p.add_argument('--limit',type=int,default=10,help='0 means all selected')
    p.add_argument('--dry-run',action='store_true',help='Preview without loading the LLM')
    p.add_argument('--resume',action='store_true',help='Skip previously saved question IDs')
    p.add_argument('--overwrite',action='store_true',help='Replace existing suggestion file')
    p.add_argument('--max-input-tokens',type=int,default=900)
    p.add_argument('--max-new-tokens',type=int,default=512)
    args=p.parse_args(argv)
    if args.limit<0 or args.max_input_tokens<128 or args.max_new_tokens<32:
        p.error('Invalid token or limit setting')
    return run(args)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (RuntimeError,ValueError,FileNotFoundError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
