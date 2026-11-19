import importlib.util
import unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location('qa_format', Path(__file__).resolve().parents[1]/'dataset_engineering'/'ai_assisted_review.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class FormatTests(unittest.TestCase):
    def test_labeled(self):
        p=mod.parse_suggestion('QUESTION: What is unit testing?\nANSWER: It tests isolated components.\nNOTES: Verify wording.')
        self.assertEqual('SUGGESTED',mod.suggestion_status(p))
    def test_json_alias(self):
        p=mod.parse_suggestion('{"question":"What is API?", "answer":"An interface between programs."}')
        self.assertEqual('What is API?',p['suggested_question'])
    def test_json_canonical(self):
        p=mod.parse_suggestion('{"suggested_question":"What is Git?", "suggested_answer":"A version control system.", "ai_review_notes":"check source"}')
        self.assertEqual('SUGGESTED',mod.suggestion_status(p))
    def test_reject_keyvalue(self):
        with self.assertRaisesRegex(ValueError,'question/answer pair'):
            mod.parse_suggestion('{"key":"question", "value":"What does the source explain?"}')
    def test_reject_malformed(self):
        with self.assertRaises(ValueError):
            mod.parse_suggestion('{"suggested_question": "Why?", "suggested_answer": "A "quote""}')
    def test_inadequate_evidence(self):
        p=mod.parse_suggestion('QUESTION: What is a test?\nANSWER: INSUFFICIENT_EVIDENCE\nNOTES: No relevant evidence.')
        self.assertEqual('NEEDS_MANUAL_REVIEW',mod.suggestion_status(p))
    def test_template_not_suggested(self):
        p=mod.parse_suggestion('QUESTION: What does the source explain about Git?\nANSWER: It is a VCS.\nNOTES: check')
        self.assertEqual('NEEDS_MANUAL_REVIEW',mod.suggestion_status(p))
    def test_missing_answer(self):
        with self.assertRaises(ValueError):mod.parse_suggestion('QUESTION: What is Git?\nNOTES: check')
