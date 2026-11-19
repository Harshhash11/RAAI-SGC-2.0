import csv,json,tempfile,unittest
from pathlib import Path
from training.validation import validate_training_provenance
from evaluation.prepare_answer_review import create

class TrainingProvenanceTests(unittest.TestCase):
    def _fixtures(self,root):
        root=Path(root)
        approved=root/'approved.csv';master=root/'master.csv'
        with approved.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['record_type','review_status','license_status','split','source_id','question','answer_draft']);w.writeheader()
            w.writerow(dict(record_type='QA_DRAFT',review_status='APPROVED',license_status='APPROVED',split='train',source_id='s1',question='Q?',answer_draft='A.'))
        with master.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['record_type','source_id','split']);w.writeheader();w.writerow(dict(record_type='SOURCE',source_id='s1',split='train'))
        row=dict(source_id='s1',messages=[{'role':'system','content':'IT expert'}, {'role':'user','content':'Q?'},{'role':'assistant','content':'A.'}])
        return approved,master,row
    def test_matching_approved_source(self):
        with tempfile.TemporaryDirectory() as d:
            app,master,row=self._fixtures(d)
            self.assertEqual(validate_training_provenance([row],app,master)['validated_examples'],1)
    def test_rejects_invented_answers(self):
        with tempfile.TemporaryDirectory() as d:
            app,master,row=self._fixtures(d);row['messages'][2]['content']='Invented.'
            with self.assertRaises(ValueError):validate_training_provenance([row],app,master)
    def test_rejects_test_split(self):
        with tempfile.TemporaryDirectory() as d:
            app,master,row=self._fixtures(d)
            text=master.read_text().replace('s1,train','s1,test');master.write_text(text)
            with self.assertRaises(ValueError):validate_training_provenance([row],app,master)
    def test_review_sheet_has_blank_ratings(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pred.json';out=Path(d)/'review.csv';path.write_text(json.dumps([dict(question='Q?',answer='A.',source_ids=['s1'])]))
            create(path,out)
            r=list(csv.DictReader(out.open()))[0]
            self.assertEqual(r['factual_correctness_1_to_5'],'')
            self.assertEqual(r['answer'],'A.')
if __name__=='__main__':unittest.main()
