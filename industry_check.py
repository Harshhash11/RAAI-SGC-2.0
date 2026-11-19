"""Fast, offline checks. Does not load or download model weights."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent

def main():
    ok = True
    def check(name, passed, detail=''):
        nonlocal ok
        ok &= bool(passed)
        print(('PASS' if passed else 'FAIL') + ' | ' + name + (f' | {detail}' if detail else ''))
    master = ROOT/'data/processed/master.csv'
    check('Master dataset', master.is_file())
    if master.is_file():
        with master.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            check('Dataset columns', all(x in (reader.fieldnames or []) for x in ('record_type','source_id')))
    adapter=ROOT/'models/checkpoints/qlora_final_44'
    check('Fine-tuned adapter configuration', (adapter/'adapter_config.json').is_file())
    check('Fine-tuned adapter weights', (adapter/'adapter_model.safetensors').is_file())
    if (adapter/'adapter_config.json').is_file():
        try:
            cfg=json.loads((adapter/'adapter_config.json').read_text())
            check('Adapter base model', bool(cfg.get('base_model_name_or_path')))
        except Exception as e:
            check('Adapter config readable',False,str(e))
    train=ROOT/'data/training/sft_train.jsonl'
    check('Prepared approved training examples', train.is_file())
    if train.is_file():
        with train.open(encoding='utf-8') as f:
            count=sum(bool(line.strip()) for line in f)
        check('At least 20 reviewed training records',count >= 20, f'{count} records')
    try:
        from app.gradio_app import resolve_local_model, format_answer_with_evidence, FINE_TUNED_MODEL
        check('App fine-tuned selection', resolve_local_model('Local LLM', FINE_TUNED_MODEL)==str(adapter))
        text=format_answer_with_evidence({'answer':'Example answer','citations':[]},[{'source_title':'Example','source_url':'https://example.com'}])
        check('Retrieved source display', 'Retrieved source documents' in text)
    except Exception as e:
        check('Application imports and selectors', False, f'{type(e).__name__}: {e}')
    print('\n' + ('OFFLINE CHECKS PASSED' if ok else 'OFFLINE CHECKS NEED ATTENTION'))
    print('These checks do not test GPU inference, graph rendering, or factual accuracy.')
    return 0 if ok else 1

if __name__=='__main__':
    sys.exit(main())
