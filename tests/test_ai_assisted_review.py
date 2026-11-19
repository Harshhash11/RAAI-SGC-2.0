import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1]/'dataset_engineering'/'ai_assisted_review.py'
spec = importlib.util.spec_from_file_location('qa_ai_review_patch',MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class QAAssistedReviewTests(unittest.TestCase):
    def row(self):
        return {'question_id':'qa:x','source_id':'s1','split':'train','question_type':'technical_context',
                'question':'What does the source explain about x?',
                'answer_draft':'Answer','evidence_quote':'Evidence X','quality_flags':'TEMPLATE_CONTEXT_QUESTION'}
    def test_flags(self):
        self.assertIn('TEMPLATE_CONTEXT_QUESTION', mod.flags(self.row()))
    def test_parse_json(self):
        self.assertEqual('A?',mod.parse_json('```json\n{"suggested_question":"A?","suggested_answer":"B","ai_review_notes":"C"}\n```')['suggested_question'])
        with self.assertRaises(ValueError):mod.parse_json('not json')
    def test_base_unapproved(self):
        r=mod.base_record(self.row(),'qwen')
        self.assertEqual('PENDING',r['human_review_status'])
        self.assertEqual('NOT_RUN',r['generation_status'])
    def test_atomic_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'output.csv';mod.write_csv_atomic(p,[mod.base_record(self.row(),'qwen')])
            with p.open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(1,len(rows));self.assertEqual('PENDING',rows[0]['human_review_status'])
    def test_selection_train_and_flagged(self):
        a=self.row(); b=dict(a,question_id='qa:y',split='test')
        c=dict(a,question_id='qa:z',question='What is x?',question_type='definition',quality_flags='')
        self.assertEqual(['qa:x'],[x['question_id'] for x in mod.choose([a,b,c],'train',True,0)])
    def test_evidence_required(self):
        a=self.row(); a['evidence_quote']=''
        self.assertIn('MISSING_EVIDENCE',mod.flags(a))
if __name__=='__main__': unittest.main()
