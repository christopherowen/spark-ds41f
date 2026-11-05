"""Out-of-band checks for the no-loss decision, not serving instrumentation."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from check_quality import canonical_check,check


def fixture(benchmark):
    generations={};grades={}
    for label in ('native-a','native-b','lsa-a','lsa-b'):
        rows=[{'id':'one','prompt_sha256':'p','status':'complete','finish_reason':'stop'}]
        generations[label]={'complete':True,'records':rows,'source_lock':{'protocol':'shared'},'manifest_sha256':'frozen'}
        grades[label]={'complete':True,'records':[{'id':'one','correct':True,'score':100}]}
    return generations,grades


class GateTests(unittest.TestCase):
    def test_longbench_adapter_uses_its_phase_gate(self):
        spec=importlib.util.spec_from_file_location('matrix_longbench',Path(__file__).with_name('run_window.py'))
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        root=Path(__file__).parent.parent/'2026-11-05-query-reuse/runs/pilot-window-02'
        runs={label:json.loads((root/('answers-'+label+'.json')).read_text())
              for label in ('native-a','native-b','lsa-a','lsa-b')}
        self.assertTrue(controller.quality_result(runs,runs,'longbench')['passed'])
        self.assertTrue(controller.early_result(runs,runs,'lsa-a','longbench')['passed'])
        changed=copy.deepcopy(runs)
        row=next(r for r in changed['lsa-a']['phases'][0]['records'] if r['correct'])
        row.update(correct=False,status='truncated',finish_reason='length')
        self.assertFalse(controller.early_result(changed,changed,'lsa-a','longbench')['passed'])
        self.assertFalse(controller.quality_result(changed,changed,'longbench')['passed'])

    def test_custom_profile_does_not_change_restoration_authority(self):
        spec=importlib.util.spec_from_file_location('matrix_profile',Path(__file__).with_name('run_window.py'))
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        original=controller.w.REL
        config=Path('experiments/2026-11-05-compact-runtime/candidate.json')
        command=controller.profile_action(config,'doctor')
        self.assertIn(str(config),command)
        self.assertEqual(controller.w.REL,original)
        with patch.object(controller.w,'start',side_effect=RuntimeError('boot failed')):
            with self.assertRaises(RuntimeError):controller.profile_action(config,'start',None,'boot')
        self.assertEqual(controller.w.REL,original)

    def test_recorded_reference_failure_does_not_omit_model_tests(self):
        path=Path(__file__).parent/'runs/code-window-03/grade-oracle/receipt.json'
        receipt=json.loads(path.read_text())
        result=canonical_check(receipt)
        self.assertEqual(result['known_reference_failures'],['HumanEval/32'])
        self.assertEqual(result['model_tasks_omitted'],[])
        self.assertFalse(result['model_tests_changed'])
        changed=copy.deepcopy(receipt);changed['records'][0]['correct']=False
        with self.assertRaises(AssertionError):canonical_check(changed)
        changed=copy.deepcopy(receipt)
        next(r for r in changed['records'] if r['id']=='HumanEval/32')['plus']['passed_tests']-=1
        with self.assertRaises(AssertionError):canonical_check(changed)

    def test_controller_reaches_read_only_identity_preflight(self):
        spec=importlib.util.spec_from_file_location('matrix_preflight',Path(__file__).with_name('run_window.py'))
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder).resolve();cache=home/'cache';cache.mkdir()
            out=home/'spark-lab/results/test'
            with patch.object(controller.socket,'gethostname',return_value='spark-0009'), \
                 patch.object(controller.subprocess,'check_output',side_effect=['published\n','','origin/branch\n']), \
                 patch.object(controller.w,'each',return_value={'node':{'spark-hold.json':None}}), \
                 patch.object(controller,'runtime_identity',side_effect=RuntimeError('identity preflight reached')), \
                 patch.object(Path,'home',return_value=home), \
                 patch.object(controller.sys,'argv',['window','--out',str(out),'--cache',str(cache)]):
                with self.assertRaisesRegex(RuntimeError,'identity preflight reached'):controller.main()
            self.assertFalse(out.exists())

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

    def test_candidate_can_solve_a_task_native_truncates(self):
        generations,grades=fixture('code')
        for label in ('native-a','native-b'):
            generations[label]['records'][0].update(status='incomplete',finish_reason='length')
            grades[label]['records'][0]['correct']=False
        result=check(generations,grades,'code')
        self.assertTrue(result['passed']);self.assertEqual(result['wins'],['one'])
        self.assertEqual(len(result['incomplete']),2)

    def test_shared_incomplete_is_zero_not_a_new_regression(self):
        generations,grades=fixture('code')
        for label in generations:
            generations[label]['records'][0].update(status='incomplete',finish_reason='repetition')
            grades[label]['records'][0]['correct']=False
        result=check(generations,grades,'code')
        self.assertTrue(result['passed']);self.assertEqual(len(result['incomplete']),4)
        self.assertEqual(result['new_incomplete'],[])

    def test_losing_a_completed_native_wrong_answer_still_rejects(self):
        generations,grades=fixture('code')
        for label in grades:grades[label]['records'][0]['correct']=False
        generations['lsa-a']['records'][0].update(status='incomplete',finish_reason='length')
        result=check(generations,grades,'code')
        self.assertFalse(result['passed']);self.assertEqual(result['new_incomplete'],[{'id':'one','arm':'lsa-a'}])

    def test_changed_protocol_and_identity_are_invalid(self):
        for key,value in [('source_lock',{'protocol':'different'}),('manifest_sha256','different')]:
            benchmark='code' if key=='source_lock' else 'ruler'
            generations,grades=fixture(benchmark);generations['lsa-a'][key]=value
            with self.assertRaises(AssertionError):check(generations,grades,benchmark)
        generations,grades=fixture('code');generations['lsa-a']['records'][0]['prompt_sha256']='different'
        with self.assertRaises(AssertionError):check(generations,grades,'code')


if __name__=='__main__':unittest.main()
