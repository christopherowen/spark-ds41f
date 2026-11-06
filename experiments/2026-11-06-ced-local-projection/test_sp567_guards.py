"""CPU-only lease and recovery failure-path checks; no host action is executed."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

path=Path(__file__).with_name('sp567_cohort.py')
spec=importlib.util.spec_from_file_location('cohort',path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def rows(token='owned'):
    return {node:{'spark-hold.json':{'token':token,'holder':c.HOLDER,'nodes':c.NAMES},
                  'spark3-hold.json':None,'spark-request.json':None,'spark3-request.json':None}
            for node in c.NAMES}


class Guards(unittest.TestCase):
    def test_matching_owner_is_required_on_every_rank(self):
        value=rows()
        with patch.object(c,'holds',return_value=value):
            self.assertTrue(c.owns('owned'))
            value['sp7']['spark-hold.json']['token']='foreign'
            self.assertFalse(c.owns('owned'))

    def test_legacy_hold_and_wrong_scope_are_refused(self):
        for mutate in (lambda v:v['sp5'].update({'spark3-hold.json':{'holder':'foreign'}}),
                       lambda v:v['sp5']['spark-hold.json'].update({'nodes':['sp5']})):
            value=rows();mutate(value)
            with patch.object(c,'holds',return_value=value):self.assertFalse(c.owns('owned'))

    def test_global_scope_requires_exact_holder_token_and_recovery_targets(self):
        c.GLOBAL_HOST='verified-address'
        h={'holder':'budget','window':'window','nodes':['dgx1','dgx2','dgx3']}
        h['resource_scope']={'kind':'nodes','nodes':h['nodes'],'holder':'budget','token':'window',
                             'restore':{'nodes':h['nodes']},'watchdog':{'nodes':h['nodes']}}
        with patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(h),'')):
            self.assertEqual(c.global_scope(),h)
        for bad in ({'holder':'unscoped'},dict(h,resource_scope=dict(h['resource_scope'],token='other')),
                    dict(h,resource_scope=dict(h['resource_scope'],restore={'nodes':['sp5']}))):
            with patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(bad),'')):
                with self.assertRaises(AssertionError):c.global_scope()

    def test_restore_does_not_act_on_foreign_lease(self):
        with tempfile.TemporaryDirectory() as name, patch.object(c,'owns',return_value=False):
            with self.assertRaises(AssertionError):c.restore('owned',Path(name))

    def test_independent_identity_is_checked_before_release(self):
        events=[]
        def each(program,data=None):
            if program==c.PHASE:return {'phase':True}
            if program==c.CLEANUP:return {}
            return {'sp5':{'commit':c.RESTORE_COMMIT,'dirty':''}}
        def recover(token,out,stem,finish_heartbeat=None):
            events.append('native_recovery_verified')
            finish_heartbeat()
            events.append('release')
        fake=types.SimpleNamespace(each=each,restore=recover)
        with tempfile.TemporaryDirectory() as name, patch.object(c,'owns',return_value=True), \
             patch.object(c,'recovery',fake), patch.object(c,'identity',side_effect=lambda:events.append('identity') or {}):
            c.restore('owned',Path(name),lambda:events.append('heartbeat_join'))
        self.assertEqual(events,['native_recovery_verified','identity','heartbeat_join','release'])

    def test_restore_failure_is_retained_and_does_not_release(self):
        fake=types.SimpleNamespace(each=lambda *a,**k: {'sp5':{'commit':c.RESTORE_COMMIT,'dirty':''}},
                                   restore=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('failed startup')))
        with tempfile.TemporaryDirectory() as name, patch.object(c,'owns',return_value=True),patch.object(c,'recovery',fake):
            with self.assertRaises(RuntimeError):c.restore('owned',Path(name))
            self.assertTrue((Path(name)/'restore-failed.txt').exists())
            self.assertFalse((Path(name)/'restored.json').exists())


if __name__=='__main__':unittest.main()
