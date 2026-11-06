"""CPU-only tests for the exact smoke wrapper and image-copy ownership guards."""
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

EXP=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('distribution',EXP/'distribute_sp567.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class Guards(unittest.TestCase):
    def wrapper(self,hold_token='owned',argv=None):
        with tempfile.TemporaryDirectory() as name:
            home=Path(name);(home/'spark-hold.json').write_text(json.dumps({'holder':d.c.HOLDER,'token':hold_token,'nodes':d.c.NAMES,'phase':'native'}))
            argv=argv or ['docker','run','--rm','--memory=16g','--memory-swap=16g','candidate','-']
            with patch.object(Path,'home',return_value=home),patch.object(sys,'argv',argv), \
                 patch.dict(os.environ,{'CED_SMOKE_TOKEN':'owned','CED_SMOKE_NAME':'ced-sp567-owned-smoke','CED_SMOKE_IMAGE':'candidate'}), \
                 patch.object(os,'execv',side_effect=SystemExit(0)) as execute:
                with self.assertRaises((SystemExit,AssertionError)) as result:runpy.run_path(str(EXP/'docker-owned/docker'),run_name='__main__')
                return result.exception,execute.call_args
    def test_standard_smoke_keeps_limits_and_adds_exact_ownership(self):
        stopped,call=self.wrapper();self.assertIsInstance(stopped,SystemExit)
        command=call.args[1];self.assertIn('--memory=16g',command);self.assertIn('--cpus=1',command)
        self.assertIn('ced-sp567-owned-smoke',command);self.assertIn('ced.window=owned',command)
    def test_foreign_smoke_holder_never_executes_docker(self):
        stopped,call=self.wrapper(hold_token='foreign');self.assertIsInstance(stopped,AssertionError);self.assertIsNone(call)
    def test_read_only_docker_commands_are_unchanged(self):
        stopped,call=self.wrapper(argv=['docker','ps','--format','{{.Names}}']);self.assertIsInstance(stopped,SystemExit)
        self.assertEqual(call.args,('/usr/bin/docker',['docker','ps','--format','{{.Names}}']))
    def test_copy_requires_every_owned_node_memory_and_fresh_global_read(self):
        base={'hold':{'holder':d.c.HOLDER,'token':'owned','nodes':d.c.NAMES,'phase':'copying'},'legacy':False,'memory':100*2**30}
        states={node:copy.deepcopy(base) for node in d.NODES}
        observed={'source':'dgx1:~/spark-hold.json','success':True,'observed_at':d.c.stamp(),'read_duration_seconds':0.5,'hold':None}
        def remote(node,program):return states[node] if program==d.READ else observed
        with patch.object(d.c,'remote',side_effect=remote),patch.object(d.spark,'available_memory_gib',return_value=100):
            d.guard('owned')
            for key,value in [('token','foreign'),('phase','native')]:
                states[d.NODES[2]]['hold'][key]=value
                with self.assertRaises(AssertionError):d.guard('owned')
                states[d.NODES[2]]=copy.deepcopy(base)
            states[d.NODES[2]]['memory']=47*2**30
            with self.assertRaises(AssertionError):d.guard('owned')
            states[d.NODES[2]]=copy.deepcopy(base);observed['success']=False
            with self.assertRaises(AssertionError):d.guard('owned')

if __name__=='__main__':unittest.main()
