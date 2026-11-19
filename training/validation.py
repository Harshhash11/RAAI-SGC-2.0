"""Validate source/provenance and split isolation before supervised training.

Approval cannot be granted by this script. It only cross-checks separately
reviewed export files. Reviewers must supply human decisions first.
"""
import csv
from pathlib import Path


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def validate_training_provenance(rows, approved_csv, master_csv):
    approved = read_csv(approved_csv)
    sources = {r['source_id']: r for r in read_csv(master_csv) if r['record_type']=='SOURCE'}
    keys = {}
    for r in approved:
        if r['record_type'] != 'QA_DRAFT' or r['review_status'] != 'APPROVED' or r['license_status'] != 'APPROVED':
            raise ValueError('Approved export contains an unapproved/invalid row')
        if r['split'] != 'train':
            continue
        key = (r['source_id'], r['question'].strip(), r['answer_draft'].strip())
        keys[key] = r
    if not keys:
        raise ValueError('No human-approved, train-split examples in approved export')
    seen=set()
    for n, row in enumerate(rows,1):
        source=row['source_id']
        msgs=row['messages']
        key=(source,msgs[1]['content'].strip(),msgs[2]['content'].strip())
        if key not in keys:
            raise ValueError(f'Training row {n} is absent from approved export')
        if key in seen:
            raise ValueError(f'Duplicate training example at row {n}')
        seen.add(key)
        if source not in sources or sources[source]['split'] != 'train':
            raise ValueError(f'Training row {n} uses missing/held-out source')
    return {'validated_examples':len(rows),'approved_train_records':len(keys), 'verified_against_export':True}
