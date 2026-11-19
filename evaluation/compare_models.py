"""RAAI-SGC held-out prediction export (not an automatic quality evaluator).

Run each configuration in a fresh Python process to keep GPU memory usage low.
Test questions are unverified drafts until independently reviewed.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = {
    'baseline': ('vector', 'Qwen/Qwen2.5-1.5B-Instruct'),
    'graph': ('hybrid', 'Qwen/Qwen2.5-1.5B-Instruct'),
    'finetuned': ('hybrid', 'models/checkpoints/qlora_final_44'),
}


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=CONFIGS, required=True)
    parser.add_argument('--limit', type=int, default=5,
                        help='Number of test questions; 0 means all. Default 5 for smoke test.')
    parser.add_argument('--master', type=Path, default=ROOT / 'data/processed/master.csv')
    parser.add_argument('--approved', type=Path, default=ROOT / 'data/review/APPROVED_QA_ONLY.csv')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'data/evaluation')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')

    rows = read_rows(args.master)
    required = {'record_type', 'split', 'source_id', 'question_id', 'question'}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise SystemExit(f'Missing master.csv fields: {sorted(missing)}')

    test = [r for r in rows if r['record_type'] == 'QA_DRAFT'
            and r['split'].strip().lower() == 'test'
            and r['question'].strip()]
    if not test:
        raise SystemExit('No test-split QA_DRAFT questions found; refusing to use training questions.')

    approved = read_rows(args.approved)
    train_sources = {r.get('source_id', '') for r in approved}
    overlap = {r['source_id'] for r in test} & train_sources
    if overlap:
        raise SystemExit(f'Data leakage: {len(overlap)} test source IDs overlap approved examples: {sorted(overlap)[:5]}')
    if len({r['question_id'] for r in test}) != len(test):
        raise SystemExit('Duplicate test question IDs; cannot compare reliably.')
    test = test[:args.limit] if args.limit else test

    out = args.outdir / f'predictions_{args.config}.csv'
    if out.exists():
        raise SystemExit(f'Output already exists: {out}. Rename or move it before rerunning.')

    mode, model_id = CONFIGS[args.config]
    if args.config == 'finetuned':
        adapter = ROOT / model_id
        if not (adapter / 'adapter_config.json').is_file():
            raise SystemExit(f'Fine-tuned adapter not found: {adapter}')
        model_id = str(adapter)

    # Imports delayed until the test/provenance checks have passed.
    from src.hybrid import HybridRetriever, load_data
    from src.llm import generate, load_generator
    sources, edges, _ = load_data(args.master)
    retriever = HybridRetriever(sources, edges, backend='tfidf')

    print(f'Configuration: {args.config} | retrieval={mode} | test questions={len(test)}', flush=True)
    print('Loading model (excluded from per-question latency)...', flush=True)
    load_generator(model_id)

    output = []
    for idx, record in enumerate(test, 1):
        question = record['question'].strip()
        start = time.perf_counter()
        hits, diagnostics = retriever.search(question, top_k=3, mode=mode)
        answer = generate(question, hits, model_id, max_new_tokens=180)
        elapsed = round(time.perf_counter() - start, 3)
        hit_meta = []
        for h in hits:
            hit_meta.append({k: h.get(k) for k in (
                'source_id', 'source_title', 'vector_score', 'graph_score', 'source_url'
            ) if k in h})
        output.append({
            'config': args.config,
            'question_id': record['question_id'],
            'source_id': record['source_id'],
            'question': question,
            'answer': answer.get('answer', ''),
            'retrieved_sources_json': json.dumps(hit_meta, ensure_ascii=False),
            'citations_json': json.dumps(answer.get('citations', []), ensure_ascii=False),
            'graph_activated': str(diagnostics.get('graph_activated', False)),
            'latency_seconds': elapsed,
            'human_correctness_1to5': '',
            'human_evidence_support_1to5': '',
            'human_relevance_1to5': '',
            'reviewer': '',
            'review_notes': '',
        })
        print(f'[{idx}/{len(test)}] {record["question_id"]} | {elapsed:.2f}s', flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print('Saved:', out)
    print('NOTE: Unreviewed test questions are not established ground truth. Human scores remain blank.')


if __name__ == '__main__':
    main()
