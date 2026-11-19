import argparse
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH=Path(__file__).resolve().parents[1]/'dataset_engineering/ai_assisted_review.py'
spec=importlib.util.spec_from_file_location('qa_review_retry',MODULE_PATH)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class RetryTests(unittest.TestCase):
    def row(self):
        return dict(question_id='q1',source_id='s1',split='train',question_type='technical_context',question='What does the source explain about x?',answer_draft='Draft',evidence_quote='x is a system')
    def test_retry_replaces_error_without_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); inp=d/'input.csv';out=d/'out.csv'
            with inp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=self.row());w.writeheader();w.writerow(self.row())
            old=mod.base_record(self.row(),mod.MODEL_ID);old.update(generation_status='GENERATION_ERROR',ai_review_notes='Invalid JSON')
            mod.write_csv_atomic(out,[old])
            args=argparse.Namespace(input=inp,output=out,split='train',all_questions=False,limit=10,dry_run=False,resume=True,overwrite=False,model=mod.MODEL_ID,max_input_tokens=900,max_new_tokens=512)
            with patch.object(mod,'load_model',return_value=('tok','model')) as load, patch.object(mod,'generate',return_value=dict(suggested_question='What is x?',suggested_answer='x is a system.',ai_review_notes='Needs verification')):
                mod.run(args)
            with out.open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(1,len(rows));self.assertEqual('SUGGESTED',rows[0]['generation_status']);self.assertEqual('PENDING',rows[0]['human_review_status'])
            load.assert_called_once()
    def test_resume_preserves_success_and_skips_model(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); inp=d/'in.csv'; out=d/'out.csv'
            with inp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=self.row());w.writeheader();w.writerow(self.row())
            old=mod.base_record(self.row(),mod.MODEL_ID);old['generation_status']='SUGGESTED';mod.write_csv_atomic(out,[old])
            args=argparse.Namespace(input=inp,output=out,split='train',all_questions=False,limit=10,dry_run=False,resume=True,overwrite=False,model=mod.MODEL_ID,max_input_tokens=900,max_new_tokens=512)
            with patch.object(mod,'load_model',side_effect=AssertionError('must not load')):mod.run(args)
            with out.open() as f:self.assertEqual(1,len(list(csv.DictReader(f))))

if __name__=='__main__':unittest.main()
