import unittest
from pathlib import Path
from src.grounding import cited_sources

class AdapterAndCitationsTests(unittest.TestCase):
    def test_no_unclaimed_citations(self):
        rows=[{'source_title':'A','source_id':'a','source_url':'https://x/a'}, {'source_title':'B','source_id':'b','source_url':'https://x/b'}]
        self.assertEqual(cited_sources('answer [2]',rows)[0]['source_id'],'b')
        self.assertEqual(cited_sources('no bracket evidence',rows),[])
    def test_project_adapter_absent_not_hidden(self):
        self.assertFalse((Path(__file__).resolve().parents[1]/'models/checkpoints/colab_qlora/adapter_config.json').exists())
