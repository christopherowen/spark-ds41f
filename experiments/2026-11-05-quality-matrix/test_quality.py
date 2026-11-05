"""Out-of-band checks for the no-loss decision, not serving instrumentation."""
import copy
import unittest
from check_quality import check


def fixture(benchmark):
    generations={};grades={}
    for label in ('native-a','native-b','lsa-a','lsa-b'):
        rows=[{'id':'one','prompt_sha256':'p','status':'complete','finish_reason':'stop'}]
        generations[label]={'complete':True,'records':rows,'source_lock':{'protocol':'shared'},'manifest_sha256':'frozen'}
        grades[label]={'complete':True,'records':[{'id':'one','correct':True,'score':100}]}
    return generations,grades


class GateTests(unittest.TestCase):
    def test_both_benchmarks_accept_unchanged_scores(self):
        for benchmark in ('code','ruler'):
            self.assertTrue(check(*fixture(benchmark),benchmark)['passed'])

    def test_fractional_target_loss_rejects(self):
        generations,grades=fixture('ruler');grades['lsa-b']['records'][0]['score']=99
        result=check(generations,grades,'ruler')
        self.assertFalse(result['passed']);self.assertEqual(result['losses'],['one'])

    def test_one_boot_code_regression_rejects(self):
        generations,grades=fixture('code');grades['lsa-a']['records'][0]['correct']=False
        self.assertFalse(check(generations,grades,'code')['passed'])

    def test_incomplete_cannot_pass_with_correct_grade(self):
        generations,grades=fixture('code');generations['lsa-a']['records'][0]['finish_reason']='length'
        self.assertFalse(check(generations,grades,'code')['passed'])

    def test_changed_protocol_and_identity_are_invalid(self):
        for key,value in [('source_lock',{'protocol':'different'}),('manifest_sha256','different')]:
            benchmark='code' if key=='source_lock' else 'ruler'
            generations,grades=fixture(benchmark);generations['lsa-a'][key]=value
            with self.assertRaises(AssertionError):check(generations,grades,benchmark)
        generations,grades=fixture('code');generations['lsa-a']['records'][0]['prompt_sha256']='different'
        with self.assertRaises(AssertionError):check(generations,grades,'code')


if __name__=='__main__':unittest.main()
