import unittest
from src.grounding import clean_excerpt,has_evidence,evidence_messages,cited_sources
from src.llm import extract_generated,generate
from evaluation.compare_retrievers import approved_gold
from pathlib import Path
from tempfile import TemporaryDirectory

class GroundingTests(unittest.TestCase):
 def setUp(self):
  self.r=[{'source_title':'Kubernetes','source_url':'https://en.wikipedia.org/wiki/Kubernetes',
           'source_text':'Kubernetes orchestrates containers.','source_id':'s1','vector_score':0.23}]
 def test_clean(self): self.assertEqual(clean_excerpt('Hello ()  world [3]'), 'Hello world')
 def test_low_evidence(self): self.assertFalse(has_evidence([dict(self.r[0],vector_score=0)]))
 def test_prompt(self):
  x=evidence_messages('What is Kubernetes?',self.r)
  self.assertEqual(x[0]['role'],'system');self.assertIn('Kubernetes',x[1]['content'])
 def test_citations(self):
  self.assertEqual(len(cited_sources('Answer [1].',self.r)),1)
  self.assertFalse(cited_sources('No citations',self.r))
 def test_no_model_download_without_evidence(self):
  x=generate('unknown',[dict(self.r[0],vector_score=0)],'unavailable-model')
  self.assertEqual(x['citations'],[])
 def test_eval_requires_gold_approval(self):
  with TemporaryDirectory() as tmp:
   p=Path(tmp)/'independent.csv'
   p.write_text('evaluation_id,question,evidence_source_id,review_status,split\n1,What is Kubernetes?,s1,DRAFT,test\n')
   with self.assertRaises(ValueError): approved_gold(p,{'s1':{}})
 def test_generated_text(self):
  self.assertEqual(extract_generated([{'generated_text':' yes '}]),'yes')
  self.assertEqual(extract_generated([{'generated_text':[{'role':'assistant','content':' yes '}]}]),'yes')
if __name__=='__main__':unittest.main()
