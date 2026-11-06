import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"experiments/2026-11-04-lossless-memory/scoped_window.py"
spec=importlib.util.spec_from_file_location("lossless_scoped_window",PATH)
scope=importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)

class ScopedLeaseTests(unittest.TestCase):
    def lease(self):
        return {"holder":scope.HOLDER,"window":"token","production_config":scope.RESTORE,
                "nodes":["dgx1","dgx2","dgx3"],
                "resource_scope":{"kind":"nodes","holder":scope.HOLDER,"token":"token",
                    "nodes":["dgx1","dgx2","dgx3"],
                    "restore":{"profile":scope.RESTORE,"nodes":["dgx1","dgx2","dgx3"]}}}

    def test_complete_exact_lease_is_owned(self):
        self.assertTrue(scope.owns(self.lease(),"token"))

    def test_foreign_holder_or_token_is_rejected(self):
        for key,value in [("holder","another owner"),("window","other token")]:
            h=self.lease();h[key]=value
            self.assertFalse(scope.owns(h,"token"))
        h=self.lease();h["resource_scope"]["token"]="other token"
        self.assertFalse(scope.owns(h,"token"))

    def test_missing_or_overlapping_node_scope_is_rejected(self):
        for nodes in [[],None,["dgx1","dgx2"],["dgx1","dgx2","dgx3","sp5"],
                      ["dgx1","dgx2","sp5"],["dgx1","dgx2","dgx3","dgx3"],[{},None,1]]:
            for field in ("nodes","resource_scope"):
                h=self.lease()
                if field=="nodes":h["nodes"]=nodes
                else:h["resource_scope"]["nodes"]=nodes
                self.assertFalse(scope.owns(h,"token"))

    def test_restore_group_and_profile_are_part_of_lease(self):
        h=self.lease();h["resource_scope"]["restore"]["nodes"]=["sp5","sp6","sp7"]
        self.assertFalse(scope.owns(h,"token"))
        h=self.lease();h["resource_scope"]["restore"]["profile"]="another.json"
        self.assertFalse(scope.owns(h,"token"))

    def test_absent_or_malformed_scope_is_not_owned(self):
        for h in [None,[],{},{"resource_scope":[]},{"resource_scope":{"restore":None}}]:
            self.assertFalse(scope.owns(h,"token"))

    def test_absent_node_holds_are_accepted(self):
        def run(nodes,node,*argv):
            return SimpleNamespace(returncode=0,stdout=json.dumps([
                {"name":"spark-hold.json","exists":False,"value":None},
                {"name":"spark3-hold.json","exists":False,"value":None}]),stderr="")
        rows,problems=scope.node_hold_observations(SimpleNamespace(run_ssh=run),
            {"nodes":[{"name":"dgx2"}]},"token")
        self.assertEqual(problems,[])
        self.assertEqual(rows[0]["node"],"dgx2")

    def test_foreign_or_malformed_node_hold_is_rejected(self):
        for value in [{},None,[],{"holder":"another task"},self.lease()]:
            def run(nodes,node,*argv):
                return SimpleNamespace(returncode=0,stdout=json.dumps([
                    {"name":"spark-hold.json","exists":True,"value":value},
                    {"name":"spark3-hold.json","exists":False,"value":None}]),stderr="")
            rows,problems=scope.node_hold_observations(SimpleNamespace(run_ssh=run),
                {"nodes":[{"name":"dgx2"}]},"wrong-token")
            self.assertTrue(problems)

    def test_failed_or_incomplete_node_read_is_rejected(self):
        for exit_code,text in [(255,""),(0,"not json"),(0,"[]")]:
            def run(nodes,node,*argv):
                return SimpleNamespace(returncode=exit_code,stdout=text,stderr="")
            rows,problems=scope.node_hold_observations(SimpleNamespace(run_ssh=run),
                {"nodes":[{"name":"dgx2"}]},"token")
            self.assertTrue(problems)

    def test_child_apply_flag_does_not_enable_helper(self):
        p=subprocess.run([sys.executable,str(PATH),"exec","--source-commit","0"*40,
                          "--restore-commit","1"*40,"--out","results/private/test-scope",
                          "--","not-a-command","--apply"],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue(json.loads(p.stdout)["plan_only"])

if __name__=="__main__":unittest.main()
