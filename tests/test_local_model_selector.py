import unittest
from app.gradio_app import (
    DEFAULT_LOCAL_MODEL, CUSTOM_MODEL,
    resolve_local_model, model_selector_visibility,
)

class LocalModelSelectorTests(unittest.TestCase):
    def test_qwen_selected_in_local_mode(self):
        self.assertEqual(resolve_local_model('Local LLM', DEFAULT_LOCAL_MODEL), DEFAULT_LOCAL_MODEL)

    def test_extractive_ignores_model_selection(self):
        self.assertEqual(resolve_local_model('Extractive (offline)', DEFAULT_LOCAL_MODEL), '')
        self.assertFalse(model_selector_visibility('Extractive (offline)'))

    def test_custom_adapter(self):
        self.assertEqual(resolve_local_model('Local LLM', CUSTOM_MODEL, ' models/checkpoints/adapter '),
                         'models/checkpoints/adapter')
        with self.assertRaises(ValueError):
            resolve_local_model('Local LLM', CUSTOM_MODEL)

    def test_mode_visibility(self):
        self.assertTrue(model_selector_visibility('Local LLM'))
        self.assertFalse(model_selector_visibility('Extractive (offline)'))

if __name__ == '__main__':
    unittest.main()
