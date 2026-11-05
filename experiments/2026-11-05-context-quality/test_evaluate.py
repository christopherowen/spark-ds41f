"""Final-content scoring and complete offline-input preflight checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('ruler_client_tested',Path(__file__).with_name('evaluate.py'))
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


class Response:
    def __init__(self,value):self.value=value
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):return json.dumps(self.value).encode()


class RulerClientTests(unittest.TestCase):
    def answer(self,content,reasoning,finish):
        case={'id':'case','task':'needle','family':'niah','generator_length':32768,
              'prompt':'public synthetic fixture','prompt_sha256':'fixture','prompt_tokens_untemplated':5,
              'outputs':['NEEDLE']}
        protocol={'temperature':0,'seed':0,'max_tokens':65536,'thinking':True,
                  'reasoning_effort':'high','request_timeout_s':1800}
        body={'model':'m','choices':[{'finish_reason':finish,'message':{'content':content,'reasoning':reasoning}}],
              'usage':{'prompt_tokens':5,'completion_tokens':10}}
        metric=lambda predictions,references:100 if predictions==['NEEDLE'] else 0
        with tempfile.TemporaryDirectory() as folder,patch.object(client.urllib.request,'urlopen',return_value=Response(body)):
            return client.one('http://example.invalid','m',case,protocol,metric,Path(folder))

    def test_reasoning_cannot_supply_the_final_answer(self):
        self.assertEqual(self.answer('wrong','NEEDLE','stop')['score'],0)
        self.assertEqual(self.answer('NEEDLE','different reasoning','stop')['score'],100)

    def test_partial_or_empty_final_stays_zero(self):
        self.assertEqual(self.answer('NEEDLE','reasoning','length')['score'],0)
        row=self.answer('','NEEDLE','stop')
        self.assertEqual(row['score'],0);self.assertEqual(row['status'],'incomplete')

    def test_screen_preserves_frozen_order_and_missing_grader_fails_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder);cache=home/'cache';cache.mkdir()
            rows=[{'id':f'{task}-{length}'} for task in range(13) for length in [32768,131072,524288,978944]]
            requests=cache/'requests.jsonl';requests.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            grader=cache/'grader.py';grader.write_text('# synthetic preflight fixture\n')
            manifest={'requests_sha256':client.sha(requests),'grader_sha256':client.sha(grader)}
            (cache/'evaluation.json').write_text(json.dumps(manifest))
            reference={'evaluation_sha256':client.sha(cache/'evaluation.json'),'requests_sha256':client.sha(requests),
                       'grader_sha256':client.sha(grader),'case_count':52}
            (home/'pilot.set.json').write_text(json.dumps(reference))
            ids=[r['id'] for r in rows if r['id'].endswith(('-32768','-131072'))]
            (home/'screen.set.json').write_text(json.dumps({'ids':ids,'case_count':26,
                'parent_evaluation_sha256':reference['evaluation_sha256'],
                'parent_requests_sha256':reference['requests_sha256'],'grader_sha256':reference['grader_sha256']}))
            with patch.object(client,'HERE',home):
                full=client.load_inputs(cache,'pilot');screen=client.load_inputs(cache,'pilot','screen')
                self.assertEqual(len(full[2]),52);self.assertEqual([r['id'] for r in screen[2]],ids)
                self.assertNotEqual(full[3],screen[3])
                grader.unlink()
                with self.assertRaises(FileNotFoundError):client.load_inputs(cache,'pilot','screen')


if __name__=='__main__':unittest.main()
