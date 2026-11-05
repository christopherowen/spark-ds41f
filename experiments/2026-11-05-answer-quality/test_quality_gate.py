import copy
import unittest

from check_quality import check


def report(correct=True):
    row = {'id':'document-a','prompt_sha256':'prompt','domain':'code',
           'expected':'A','predicted':'A' if correct else 'B','correct':correct,
           'status':'complete','finish_reason':'stop'}
    return {'complete':True,'split':'heldout','selection_sha256':'selection',
            'dataset_sha256':'dataset','protocol':{'max_tokens':65536},
            'phases':[{'name':'cold','concurrency':1,'cache_reuse':False,'total':1,'records':[row]}]}


class QualityGate(unittest.TestCase):
    def test_native_correct_candidate_wrong_is_rejected(self):
        result = check([report(),report()],[report(False),report(False)])
        self.assertFalse(result['passed'])
        self.assertEqual(result['phases'][0]['losses'],['document-a'])

    def test_new_correct_answer_is_allowed(self):
        result = check([report(False),report(False)],[report(),report()])
        self.assertTrue(result['passed'])
        self.assertEqual(result['phases'][0]['wins'],['document-a'])
        self.assertEqual(result['independent_documents'],1)

    def test_reasoning_and_cache_salt_are_not_quality(self):
        native,candidate = report(),report()
        native['phases'][0]['records'][0].update(reasoning_characters=1,request_sha256='salt-a')
        candidate['phases'][0]['records'][0].update(reasoning_characters=100000,request_sha256='salt-b')
        self.assertTrue(check([native,native],[candidate,candidate])['passed'])

    def test_truncation_fails_even_without_a_native_correct_answer(self):
        candidate = report(False)
        candidate['phases'][0]['records'][0].update(status='truncated',finish_reason='length',predicted=None)
        self.assertFalse(check([report(False),report(False)],[candidate,candidate])['passed'])

    def test_native_truncation_can_be_outperformed(self):
        native=report(False)
        native['phases'][0]['records'][0].update(status='truncated',finish_reason='length',predicted=None)
        result=check([native,native],[report(),report()])
        self.assertTrue(result['passed'])
        self.assertEqual(result['phases'][0]['wins'],['document-a'])
        self.assertEqual(len(result['phases'][0]['incomplete']),2)

    def test_one_bad_repeat_boot_cannot_be_hidden(self):
        result = check([report(),report()],[report(),report(False)])
        self.assertFalse(result['passed'])
        self.assertEqual(result['phases'][0]['unstable'][0]['arm'],'lsa')

    def test_missing_task_and_changed_allowance_fail_closed(self):
        candidate = report()
        candidate['phases'][0]['records'] = []
        with self.assertRaises(ValueError):check([report(),report()],[candidate,candidate])
        candidate = report();candidate['protocol']['max_tokens'] = 8192
        with self.assertRaises(ValueError):check([report(),report()],[candidate,candidate])

    def test_repeated_phases_do_not_inflate_independence(self):
        run = report();run['phases'].append(copy.deepcopy(run['phases'][0]))
        run['phases'][1]['name'] = 'cached';run['phases'][1]['cache_reuse'] = True
        self.assertEqual(check([run,run],[run,run])['independent_documents'],1)

    def test_empty_workload_is_not_a_quality_pass(self):
        run = report();run['phases'] = []
        with self.assertRaises(ValueError):check([run,run],[run,run])


if __name__ == '__main__':unittest.main()
