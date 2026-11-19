import json
import tempfile
import unittest
from pathlib import Path
from training.train_qlora_colab import load_approved, encode_example

class ToyTokenizer:
    eos_token='</s>'
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return 'SYSTEM: '+messages[0]['content']+'\nUSER: '+messages[1]['content']+'\nASSISTANT:'
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[ord(c) for c in text]}

class ColabTrainingTests(unittest.TestCase):
    def test_only_assistant_labels_contribute_to_loss(self):
        item={'messages':[{'role':'system','content':'Help'},{'role':'user','content':'Question'},{'role':'assistant','content':'Answer'}], 'source_id':'wiki:1'}
        encoded=encode_example(item,ToyTokenizer(),max_length=120)
        self.assertEqual(len(encoded['labels']),len(encoded['input_ids']))
        self.assertIn(-100,encoded['labels'])
        self.assertEqual([x for x in encoded['labels'] if x!=-100],[ord(x) for x in 'Answer</s>'])

    def test_empty_approved_data_is_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'sft.jsonl';p.write_text('')
            with self.assertRaisesRegex(RuntimeError,'Only 0 reviewed'):
                load_approved(p)

    def test_malformed_approved_data_is_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'sft.jsonl';p.write_text(json.dumps({'messages':[{'role':'user','content':'Question'}],'source_id':'wiki:1'})+'\n')
            with self.assertRaisesRegex(ValueError,'expected system/user/assistant'):
                load_approved(p)

if __name__=='__main__':unittest.main()
