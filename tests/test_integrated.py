import csv, tempfile, unittest
from pathlib import Path
from src.hybrid import load_data,HybridRetriever
from src.answers import extractive_answer
from training.prepare_sft import prepare
from evaluation.run_retrieval_eval import evaluate
ROOT=Path(__file__).resolve().parents[1]
class IntegratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources,cls.edges,cls.drafts=load_data(ROOT/'data/processed/master.csv')
    def test_source_counts(self):
        self.assertEqual(len(self.sources),145);self.assertEqual(sum(map(len,self.edges.values())),4771);self.assertEqual(self.drafts,290)
    def test_kubernetes_top_hit(self):
        r=HybridRetriever(self.sources,self.edges)
        hits,_=r.search('What is Kubernetes?',top_k=3,mode='hybrid')
        self.assertEqual(hits[0]['source_title'],'Kubernetes')
    def test_extractive_has_source(self):
        r=HybridRetriever(self.sources,self.edges)
        hits,_=r.search('What is Kubernetes?',top_k=1)
        result=extractive_answer('What is Kubernetes?',hits)
        self.assertEqual(result['mode'],'extractive');self.assertTrue(result['citations'])
    def test_no_approved_qa(self):
        with tempfile.TemporaryDirectory() as d:
            n=prepare(ROOT/'data/review/APPROVED_QA_ONLY.csv',Path(d)/'x.jsonl')
            self.assertEqual(n,0)
    def test_no_gold_eval_claim(self):
        with self.assertRaisesRegex(ValueError,'No reviewer-approved'):
            evaluate(ROOT/'data/processed/master.csv',ROOT/'data/review/INDEPENDENT_EVALUATION_TEMPLATE.csv')
if __name__=='__main__':unittest.main()
