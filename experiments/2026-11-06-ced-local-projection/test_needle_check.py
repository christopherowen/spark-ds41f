"""CPU-only negative controls for grading; tokenizer outputs are a deterministic stub."""
import array
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).with_name('check_needle.py')

class NeedleGrader(unittest.TestCase):
    def grade(self, text='{"registry":"expected-value"}', prompt=(1,2,3), error=None):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);snapshot=root/'snapshot';snapshot.mkdir()
            fixtures=root/'fixtures';fixtures.mkdir()
            raw=array.array('I',[1,2,3])
            if sys.byteorder!='little':raw.byteswap()
            manifest={'complete':True,'tokenizer_sha256':{},'max_tokens':128,
                      'cases':[{'id':'fixture','chat_token_ids_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                                'expected':{'registry':'expected-value'}}]}
            (fixtures/'manifest.json').write_text(json.dumps(manifest))
            result={'workload':{'extra_body':{'chat_template_kwargs':{'thinking':False}},'max_tokens':128},
                    'generation':{'sequences':[{'id':0,'prompt_ids':list(prompt),'output_ids':list(text.encode()),'error':error}]}}
            source=root/'result.json';source.write_text(json.dumps(result));out=root/'report.json'
            class Tokenizer:
                @classmethod
                def from_pretrained(cls,*args,**kwargs):return cls()
                def decode(self,ids,**kwargs):return bytes(ids).decode()
            fake=types.ModuleType('vllm.tokenizers.deepseek_v41');fake.DeepseekV41Tokenizer=Tokenizer
            argv=[str(SCRIPT),'--snapshot',str(snapshot),'--fixtures',str(fixtures),'--result',str(source),'--out',str(out)]
            with patch.dict(sys.modules,{'vllm.tokenizers.deepseek_v41':fake}),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:runpy.run_path(str(SCRIPT),run_name='__main__')
            return stopped.exception.code,json.loads(out.read_text())

    def test_exact_ids_and_json_pass(self):
        code,report=self.grade();self.assertEqual(code,0);self.assertTrue(report['passed'])
    def test_wrong_prompt_ids_fail_even_with_the_correct_answer(self):
        code,report=self.grade(prompt=(1,2,4));self.assertEqual(code,1)
        self.assertFalse(report['checks'][0]['prompt_ids_equal'])
    def test_wrong_value_fails(self):
        code,report=self.grade(text='{"registry":"wrong"}');self.assertEqual(code,1)
        self.assertFalse(report['checks'][0]['strict_json_exact'])
    def test_non_json_answer_fails_and_is_retained(self):
        code,report=self.grade(text='registry expected-value');self.assertEqual(code,1)
        self.assertEqual(report['checks'][0]['output_text'],'registry expected-value')
    def test_failed_model_request_is_retained(self):
        code,report=self.grade(error='request failed');self.assertEqual(code,1)
        self.assertEqual(report['checks'][0]['error'],'request failed')
    def test_empty_output_fails(self):
        code,report=self.grade(text='');self.assertEqual(code,1);self.assertFalse(report['passed'])

if __name__=='__main__':unittest.main()
