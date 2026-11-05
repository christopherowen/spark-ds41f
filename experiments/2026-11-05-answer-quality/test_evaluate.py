"""The quality oracle must use a completed final answer, never thinking text."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from prepare import select
from summarize import paired


class Response:
    def __init__(self,payload):self.payload=payload
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):return json.dumps(self.payload).encode()


class AnswerTests(unittest.TestCase):
    def run_case(self,content,reasoning='answer-A',finish='stop'):
        case={'id':'case-1','domain':'code','difficulty':'hard',
              'prompt_tokens_untemplated':65536,'prompt_sha256':'input','answer':'A'}
        response={'model':'fixed','choices':[{'message':{'content':content,'reasoning':reasoning},'finish_reason':finish}],
                  'usage':{'prompt_tokens':65550,'completion_tokens':40}}
        with tempfile.TemporaryDirectory() as directory:
            with patch('urllib.request.urlopen',return_value=Response(response)):
                receipt=evaluate.one('http://example.invalid',case,{'model':'fixed'},
                    lambda text:'A' if text == 'answer-A' else 'B' if text == 'answer-B' else None,Path(directory))
        self.assertNotIn('content',receipt)
        self.assertNotIn('reasoning',receipt)
        return receipt

    def test_correct_thinking_does_not_rescue_wrong_final(self):
        self.assertFalse(self.run_case('answer-B')['correct'])

    def test_thinking_may_change_without_penalty(self):
        self.assertTrue(self.run_case('answer-A',reasoning='entirely different thoughts')['correct'])

    def test_truncated_final_is_not_completed(self):
        row=self.run_case('answer-A',finish='length')
        self.assertFalse(row['correct']);self.assertEqual(row['status'],'truncated')

    def test_thinking_only_is_not_an_answer(self):
        row=self.run_case(None)
        self.assertFalse(row['correct']);self.assertEqual(row['status'],'missing-final')

    def test_no_fuzzy_guess_for_missing_answer_format(self):
        row=self.run_case('I discussed the possibilities.')
        self.assertFalse(row['correct']);self.assertEqual(row['status'],'unparsed-final')

    def test_superiority_counts_only_independent_task_wins_and_losses(self):
        native=[{'id':str(i),'prompt_sha256':str(i),'expected':'A','correct':False} for i in range(5)]
        lsa=[dict(r,correct=True) for r in native]
        result=paired(native,lsa)
        self.assertEqual(result['n_independent_documents'],5)
        self.assertEqual(result['accuracy_delta'],1)
        self.assertEqual(result['exact_one_sided_discordant_pair_p'],1/32)
        self.assertEqual(paired(lsa,native)['exact_one_sided_discordant_pair_p'],1)
        self.assertEqual(paired(lsa,lsa)['exact_one_sided_discordant_pair_p'],1)

    def test_frozen_selection_is_disjoint_and_order_independent(self):
        catalog=[{'id':f'{d}-{length}-{i}','domain':str(d),'prompt_tokens_untemplated':length,'context_sha256':f'context-{d}-{length}-{i}'}
                 for d in range(6) for length in (20000,100000) for i in range(5)]
        a,b=select(catalog),select(list(reversed(copy.deepcopy(catalog))))
        self.assertEqual(a['pilot'],b['pilot']);self.assertEqual(a['heldout'],b['heldout'])
        self.assertEqual(len(a['pilot']),12);self.assertEqual(len(a['heldout']),48)
        self.assertFalse(set(a['pilot']) & set(a['heldout']))
        by_id={r['id']:r for r in catalog}
        self.assertFalse({by_id[i]['context_sha256'] for i in a['pilot']} & {by_id[i]['context_sha256'] for i in a['heldout']})


if __name__ == '__main__':unittest.main()
