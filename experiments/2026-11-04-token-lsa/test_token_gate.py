"""Token identity must not silently degrade to decoded-text equivalence."""
import copy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from token_gate import capture,compare


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

    def test_concurrency_must_match(self):
        candidate = copy.deepcopy(self.record)
        candidate['concurrency'] = 4
        with self.assertRaises(ValueError):
            compare(self.record,candidate)

    def test_parallel_capture_keeps_request_response_identity(self):
        def response(request,timeout):
            payload = json.loads(request.data)
            self.assertTrue(payload['return_token_ids'])
            self.assertFalse(payload['stream'])
            token = payload['prompt'][0]
            time.sleep((8-token)*.001)
            return io.BytesIO(json.dumps({'prompt_token_ids':[token],
                'choices':[{'index':0,'token_ids':[100+token],'finish_reason':'stop'}],
                'usage':{'completion_tokens':1}}).encode())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'identity.json').write_text(json.dumps(self.record['identity']))
            (root/'requests.jsonl').write_text('\n'.join(json.dumps({'id':str(i),
                'payload':{'prompt':[i],'temperature':0}}) for i in range(8)))
            options = SimpleNamespace(identity=str(root/'identity.json'),requests=str(root/'requests.jsonl'),
                out=str(root/'capture.json'),url='http://test',timeout=1,concurrency=4)
            with patch('token_gate.urllib.request.urlopen',response):
                capture(options)
            result = json.loads((root/'capture.json').read_text())
            self.assertEqual(result['concurrency'],4)
            self.assertEqual([r['id'] for r in result['records']],[str(i) for i in range(8)])
            self.assertEqual([r['choices'][0]['token_ids'] for r in result['records']],[[100+i] for i in range(8)])

    def test_empty_suite_refused(self):
        self.record['records'] = []
        with self.assertRaises(ValueError):
            compare(self.record,self.record)


if __name__ == '__main__':
    unittest.main()
