import argparse, csv, json
from collections import Counter

def audit(path):
    types, domains, splits = Counter(), Counter(), Counter()
    q_types = Counter()
    sources = set()
    issues = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            types[row['record_type']] += 1
            if row['record_type'] == 'SOURCE':
                if row['source_id'] in sources:
                    issues.append('duplicate source: ' + row['source_id'])
                sources.add(row['source_id'])
                domains[row['domain']] += 1
                splits[row['split']] += 1
            elif row['record_type'] == 'QA_DRAFT':
                q_types[row['question_type']] += 1
    return {'record_types':dict(types),'source_domains':dict(domains),'source_splits':dict(splits),'draft_question_types':dict(q_types),'issues':issues,'approved_qa_count':0,'note':'Approved count is not inferred from drafts; requires validated approval exports.'}

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/processed/master.csv')
    args=p.parse_args()
    print(json.dumps(audit(args.csv), indent=2))
