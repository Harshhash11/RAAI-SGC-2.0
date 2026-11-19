import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from app.gradio_app import (
    DEFAULT_LOCAL_MODEL, FINE_TUNED_MODEL, CUSTOM_MODEL, FINE_TUNED_ADAPTER,
    resolve_local_model, format_answer_with_evidence, retrieved_evidence_links,
)

class IndustryHandoffTests(unittest.TestCase):
    def test_extract_mode_does_not_load_model(self):
        self.assertEqual(resolve_local_model('Extractive (offline)', FINE_TUNED_MODEL), '')
    def test_base_model(self):
        self.assertEqual(resolve_local_model('Local LLM', DEFAULT_LOCAL_MODEL), DEFAULT_LOCAL_MODEL)
    def test_custom_model(self):
        self.assertEqual(resolve_local_model('Local LLM', CUSTOM_MODEL, 'local/path'), 'local/path')
    def test_finetuned_missing_message(self):
        with patch('app.gradio_app.FINE_TUNED_ADAPTER', Path('/tmp/no_such_raai_adapter_123')):
            with self.assertRaisesRegex(ValueError, 'adapter missing'):
                resolve_local_model('Local LLM', FINE_TUNED_MODEL)
    def test_finetuned_present(self):
        with TemporaryDirectory() as d:
            p=Path(d)
            (p/'adapter_config.json').write_text('{}')
            with patch('app.gradio_app.FINE_TUNED_ADAPTER', p):
                self.assertEqual(resolve_local_model('Local LLM', FINE_TUNED_MODEL), str(p))
    def test_missing_bracket_citation_still_displays_retrieved_sources(self):
        text=format_answer_with_evidence({'answer':'Kubernetes manages containers.', 'citations':[]},
             [{'source_title':'Kubernetes', 'source_url':'https://en.wikipedia.org/wiki/Kubernetes'}])
        self.assertIn('Retrieved source documents',text)
        self.assertNotIn('Sources explicitly cited',text)
        self.assertIn('not automatically verified',text)
    def test_explicit_cited_vs_retrieved(self):
        text=format_answer_with_evidence({'answer':'A [1].','citations':[{'title':'A','url':'https://x/a'}]},
            [{'source_title':'B','source_url':'https://x/b'}])
        self.assertIn('Sources explicitly cited',text)
        self.assertIn('Retrieved source documents',text)
    def test_duplicate_and_bad_url_rejected(self):
        links=retrieved_evidence_links([{'source_title':'A','source_url':'https://a'},
            {'source_title':'A2','source_url':'https://a'},
            {'source_title':'bad','source_url':'javascript:alert(1)'}])
        self.assertEqual(links.count('https://a'),1)
        self.assertNotIn('javascript:',links)
    def test_empty(self):
        self.assertIn('No source documents', format_answer_with_evidence({'answer':'Unknown','citations':[]},[]))

if __name__=='__main__': unittest.main()
