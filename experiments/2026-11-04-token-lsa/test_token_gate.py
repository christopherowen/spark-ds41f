"""Token identity must not silently degrade to decoded-text equivalence."""
import copy
import unittest
from token_gate import compare


class TokenGateTests(unittest.TestCase):
    def setUp(self):
        self.record = {'identity':{'checkpoint':'fixed','tokenizer':'fixed'},'records':[
            {'id':'case','input_sha256':'same','completion_tokens':2,
             'choices':[{'index':0,'token_ids':[100,200],'finish_reason':'stop','stop_reason':None}]}]}

    def test_identical_tokens(self):
        self.assertTrue(compare(self.record,copy.deepcopy(self.record))['passed'])

    def test_only_tokens_and_termination_define_generated_equivalence(self):
        candidate = copy.deepcopy(self.record)
        candidate['records'][0]['choices'][0]['text'] = 'extra decoded metadata'
        self.assertTrue(compare(self.record,candidate)['passed'])

    def test_tokenized_prompt_must_match(self):
        self.record['records'][0]['prompt_token_ids'] = [1,2]
        candidate = copy.deepcopy(self.record)
        candidate['records'][0]['prompt_token_ids'] = [1,3]
        with self.assertRaises(ValueError):
            compare(self.record,candidate)

    def test_same_text_and_count_do_not_hide_changed_ids(self):
        candidate = copy.deepcopy(self.record)
        candidate['records'][0]['choices'][0].update(token_ids=[101,200],text='same')
        self.record['records'][0]['choices'][0]['text'] = 'same'
        self.assertFalse(compare(self.record,candidate)['passed'])

    def test_missing_ids_refused(self):
        del self.record['records'][0]['choices'][0]['token_ids']
        with self.assertRaises(ValueError):
            compare(self.record,self.record)

    def test_termination_differs(self):
        candidate = copy.deepcopy(self.record)
        candidate['records'][0]['choices'][0]['finish_reason'] = 'length'
        self.assertFalse(compare(self.record,candidate)['passed'])

    def test_checkpoint_and_workload_must_match(self):
        for key in ('checkpoint','tokenizer'):
            candidate = copy.deepcopy(self.record)
            candidate['identity'][key] = 'other'
            with self.assertRaises(ValueError):
                compare(self.record,candidate)
        candidate = copy.deepcopy(self.record)
        candidate['records'][0]['input_sha256'] = 'other'
        with self.assertRaises(ValueError):
            compare(self.record,candidate)

    def test_empty_suite_refused(self):
        self.record['records'] = []
        with self.assertRaises(ValueError):
            compare(self.record,self.record)


if __name__ == '__main__':
    unittest.main()
