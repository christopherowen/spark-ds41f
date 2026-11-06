"""CPU controls: hardware commands are bounded and foreign windows cannot run."""
import copy
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("collective_source_gates", HERE/"run-source-gates.py")
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)
sys.path.insert(0, str(gates.SOURCE))
from sparknet import topology
MAP = Path(os.environ.get("COLLECTIVE_PDL_MAP", HERE.parents[1]/"config/nodes.json")).resolve()

class Controls(unittest.TestCase):
    def tearDown(self):
        gates.configure("entry")

    def test_bounded_all_rank_plan(self):
        plans = gates.plans(topology.load(MAP))
        self.assertEqual([item["node"]["rank"] for item in plans], [0,1,2])
        for item in plans:
            for kind, memory, deadline in [("tests", "16g", "1500s"),("probe", "12g", "600s"),("benchmark","16g","1200s")]:
                command = item[kind]
                for flag in ["--cpus=4", "--pids-limit=512", f"--memory={memory}",
                             f"--memory-swap={memory}", deadline, gates.IMAGE,
                             f"collective-pdl.source={gates.PIN}"]:
                    self.assertIn(flag, command)
                self.assertIn(f"{gates.SOURCE}/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet:ro", command)
                self.assertFalse(any("cluster" == x for x in command))

    def test_variant_plans_use_published_identity(self):
        for phase in ("entry","staged","arrived"):
            gates.configure(phase)
            for plan in gates.plans(topology.load(MAP),run=2):
                command=plan["benchmark"]
                self.assertIn(phase,command)
                self.assertIn(gates.PIN,command)
                self.assertEqual(command[-2:],["--run","2"])
                self.assertIn(f"{gates.SOURCE}/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet:ro",command)

    def test_benchmark_requires_verified_gate_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"receipt.json"
            jobs=[]
            for kind in ("tests","probe"):
                rows=[]
                for rank in range(3):
                    log=Path(tmp)/f"{kind}-{rank}.log";log.write_text("passed\n")
                    rows.append({"node":f"dgx{rank+1}","exit":0,"problems":[],"log":log.name,
                                 "sha256":hashlib.sha256(log.read_bytes()).hexdigest()})
                jobs.append({"kind":kind,"outcomes":rows})
            data={"passed":True,"source_commit":gates.PIN,"source_tree":gates.TREE,
                  "image":gates.IMAGE,"map_sha256":"map","mode":"gates","jobs":jobs}
            path.write_text(json.dumps(data))
            self.assertEqual(gates.validate_gate(path,"map"),hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError): gates.validate_gate(path,"wrong-map")
            (Path(tmp)/"probe-0.log").write_text("changed")
            with self.assertRaises(RuntimeError): gates.validate_gate(path,"map")

    def test_probe_requires_counters_on_every_lane(self):
        from sparknet.probe.counters import RDMA_ERROR_COUNTERS
        row={"expected_backend":"oneshot","proxy_payload_bytes":{"rail0":1,"rail1":1},
             "rdma_error_deltas":{rail:{key:0 for key in RDMA_ERROR_COUNTERS} for rail in ("rail0","rail1")},
             "port_deltas":{"net0":{"tx_bytes":1,"rx_bytes":1,"rx_dropped":0}}}
        self.assertEqual(gates.probe_counter_problems([row],{"rail0","rail1"}),[])
        for changed in ("missing_error_counter","error_increase","missing_port","drop_increase","missing_payload"):
            sample=copy.deepcopy(row)
            if changed == "missing_error_counter": sample["rdma_error_deltas"]["rail1"].clear()
            elif changed == "error_increase": sample["rdma_error_deltas"]["rail0"][RDMA_ERROR_COUNTERS[0]]=1
            elif changed == "missing_port": sample["port_deltas"].clear()
            elif changed == "drop_increase": sample["port_deltas"]["net0"]["rx_dropped"]=1
            else: sample["proxy_payload_bytes"]["rail1"]=0
            with self.subTest(changed=changed):
                self.assertTrue(gates.probe_counter_problems([sample],{"rail0","rail1"}))

    def test_foreign_or_mismatched_window_cannot_touch_hardware(self):
        cases = [None,
                 {"window":"other", "holder":"spark-lab: " + gates.HOLDER_NOTE, "production_config":gates.RESTORE},
                 {"window":"ours", "holder":"spark-lab: other task", "production_config":gates.RESTORE},
                 {"window":"ours", "holder":"spark-lab: " + gates.HOLDER_NOTE, "production_config":"wrong.json"}]
        for hold in cases:
            with self.subTest(hold=hold), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/"must-not-exist"
                fake = types.ModuleType("spark_ds41f")
                fake.lab = types.SimpleNamespace(read_hold=lambda:hold)
                argv = [str(HERE/"run-source-gates.py"), "--map", str(MAP), "--out", str(out), "--window", "ours", "--apply"]
                with patch.dict(sys.modules, {"spark_ds41f":fake}), patch.object(sys, "argv", argv):
                    with self.assertRaises(RuntimeError): gates.main()
                self.assertFalse(out.exists())

if __name__ == "__main__": unittest.main()
