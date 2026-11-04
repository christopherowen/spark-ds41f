"""Native gate and failure-restoration checks without SSH, containers or GPUs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"
spec = importlib.util.spec_from_file_location("lossless_component_runner", HERE / "run_component.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.here = self.root / "experiments/memory"
        self.hold = {"holder": runner.HOLDER, "window": "owned", "production_config": runner.RESTORE}
        self.closed, self.health = [], []
        self.foreign_after_failure = False
        self.management_failure = False
        self.restore_refused = False
        self.image_failure = True
        self.plan_failure = False
        self.kernel_passes = False
        self.image = {"Id": "sha256:" + "a" * 64, "Layers": ["sha256:" + "b" * 64],
                      "Labels": {"local.spark-ds41f.vllm.tree": "c" * 40, "org.opencontainers.image.revision": "d" * 40}}
        self.images = {f"dgx{i}": self.image for i in (1, 2, 3)}
        self.nodes = {"nodes": [{"name": f"dgx{i}", "rank": i - 1, "head": i == 1} for i in (1, 2, 3)]}
        self.cluster = {"container": {"image": "candidate", "expected_labels": self.image["Labels"]}}
        self.write(self.here / "expert-scales/release-tp3.json", {"run": "release", "jobs": []})
        self.write(self.here / "expert-scales/image.json", {"run": "image", "jobs": []})
        self.write(self.root / "build/images" / ("a" * 16 + ".json"),
                   {"image_id": self.image["Id"], "tag": "candidate", "compile_jobs": 2,
                    "min_available_gib": 80, "built_utc": "fixture", "deployment_commit": "d" * 40})
        for kind in ("kernel", "full-gate"):
            bundle = self.here / (kind + "-bundle")
            self.write(bundle / "candidate.json", {"image": "candidate"})
            self.write(self.here / ("expert-scales/" + kind + ".json"),
                       {"run": kind, "jobs": [{"bundles": [{"bundle": str(bundle.relative_to(self.root))}]}]})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        if path == self.root / "hold.json":
            self.hold = value

    def close(self, reason):
        self.closed.append(reason)
        self.hold = None
        return True

    def process(self, argv, **kwargs):
        if "bin/spark" in argv and "cluster" in argv:
            self.assertNotIn("--apply", argv)
            refused = self.restore_refused and "start" in argv
            return SimpleNamespace(returncode=0, stdout="live: running: 3 nodes\nshape: unchanged\n" +
                                   ("--apply would refuse" if refused else "plan only"), stderr="")
        if "scripts/lab.py" in argv:
            kind = Path(argv[3]).stem
            if kind == "image" and self.image_failure:
                kwargs["stdout"].write("ssh: connect timed out" if self.management_failure else "build failed")
                if self.foreign_after_failure:
                    self.hold = {"holder": "spark-lab: foreign", "window": "foreign"}
                return SimpleNamespace(returncode=1)
            if kind in ("kernel", "full-gate"):
                native = {"agree": True, "verdicts": [{"node": node, "image": self.image["Id"],
                          "passed": self.kernel_passes, "exit": 0, "errors": [], "failed": []} for node in self.images]}
                self.write(self.root / "results/private/lab" / kind / "verdicts.json", native)
        return SimpleNamespace(returncode=0)

    def ssh(self, nodes, node, *argv):
        if argv == ("true",):
            self.health.append(node["name"])
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0, stdout=json.dumps(self.image))

    def invoke(self):
        lab = SimpleNamespace(read_hold=lambda: self.hold, HOLD=self.root / "hold.json", now=lambda: "fixture",
                              window_should_close=lambda hold: None, published_problems=lambda: [],
                              plan=lambda spec: [spec["run"]],
                              plan_problems=lambda steps: ["missing configuration"] if self.plan_failure else [],
                              beat=lambda note: None, write_json_atomic=self.write, window_close=self.close,
                              production_live=lambda: True)
        spark = SimpleNamespace(configuration=lambda args: (self.cluster, self.nodes, {}), run_ssh=self.ssh,
                                build_inputs=lambda upstreams: {}, build_directory=lambda inputs: self.root / "build")
        process = SimpleNamespace(run=self.process, STDOUT=runner.subprocess.STDOUT,
                                  check_output=lambda argv, **kwargs: b"" if "status" in argv else "d" * 40)
        argv = ["run_component.py", "--component", "expert-scales", "--window", "owned", "--through", "gate",
                "--out", "results/private/component", "--apply"]
        with patch.multiple(runner, ROOT=self.root, HERE=self.here, lab=lab, spark=spark, subprocess=process,
                            os=SimpleNamespace(uname=lambda: SimpleNamespace(nodename="dgx1"))), patch.object(sys, "argv", argv):
            return runner.main()

    def test_failed_build_restores_only_its_owned_window(self):
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(len(self.closed), 1)
        receipt = json.loads((self.root / "results/private/component/receipt.json").read_text())
        self.assertTrue(receipt["restored"])
        self.assertFalse(receipt["completed"])

    def test_invalid_plan_cannot_reach_an_image_or_serving_operation(self):
        self.plan_failure = True
        self.assertEqual(self.invoke(), 1)
        receipt = json.loads((self.root / "results/private/component/receipt.json").read_text())
        self.assertEqual(receipt["steps"], [])
        self.assertTrue(receipt["preflight_problems"])
        self.assertTrue(receipt["restored"])

    def test_actual_image_plan_paths_resolve_before_window_operations(self):
        for arm in ("expert-scales", "engram"):
            spec = json.loads((HERE / arm / "image.json").read_text())
            steps = runner.lab.plan(spec)
            with patch.object(runner.lab, "arm_problems", return_value=[]):
                self.assertEqual(runner.lab.plan_problems(steps), [])

    def test_foreign_window_on_entry_never_starts_or_closes(self):
        self.hold["holder"] = "foreign"
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.closed, [])
        self.assertEqual(self.health, [])

    def test_foreign_window_during_failure_is_left_alone(self):
        self.foreign_after_failure = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.closed, [])
        self.assertEqual(self.health, [])

    def test_observed_management_loss_blocks_restoration_even_after_connectivity_recovers(self):
        self.management_failure = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.closed, [])
        self.assertEqual(self.hold["holder"], "lossless-memory-hardware-review")

    def test_refused_restore_plan_keeps_a_review_hold(self):
        self.restore_refused = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.closed, [])
        self.assertEqual(self.hold["holder"], "lossless-memory-restoration-review")

    def test_failed_native_gate_cannot_start_a_serving_boot(self):
        self.image_failure = False
        self.assertEqual(self.invoke(), 1)
        receipt = json.loads((self.root / "results/private/component/receipt.json").read_text())
        self.assertFalse(receipt.get("gates_passed", False))
        self.assertFalse(any(r["log"].startswith("boot-") for r in receipt["steps"]))
        self.assertTrue(receipt["restored"])

    def test_all_three_native_ranks_can_complete_the_gate_only_stage(self):
        self.image_failure = False
        self.kernel_passes = True
        self.assertEqual(self.invoke(), 0)
        receipt = json.loads((self.root / "results/private/component/receipt.json").read_text())
        self.assertTrue(receipt["gates_passed"])
        self.assertTrue(receipt["completed"])
        self.assertTrue(receipt["restored"])

    def test_zero_exit_is_insufficient_when_native_verdict_fails_or_rank_is_missing(self):
        native = {"agree": True, "verdicts": [{"node": name, "image": self.image["Id"],
                  "passed": True, "exit": 0} for name in self.images]}
        runner.native_kernel_gate(native, self.images)
        native["verdicts"][0]["passed"] = False
        with self.assertRaises(ValueError):
            runner.native_kernel_gate(native, self.images)
        native["verdicts"].pop(0)
        with self.assertRaises(ValueError):
            runner.native_kernel_gate(native, self.images)

    def test_io_observation_requires_process_identity_and_complete_nonnegative_counters(self):
        keys = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes", "cancelled_write_bytes")
        value = {"container": "candidate", "monotonic_start_ns": 1, "monotonic_end_ns": 2,
                 "processes": {"42": {"start_ticks": 100, "io": dict.fromkeys(keys, 0)}},
                 "nvme_diskstats": {"nvme0n1": [0] * 11}}
        runner.io_gate(value, "candidate")
        for bad in ("identity", "counter", "disk", "clock"):
            row = json.loads(json.dumps(value))
            if bad == "identity":
                row["processes"]["42"]["start_ticks"] = 0
            elif bad == "counter":
                row["processes"]["42"]["io"]["read_bytes"] = -1
            elif bad == "disk":
                row["nvme_diskstats"] = {}
            else:
                row["monotonic_end_ns"] = 0
            with self.assertRaises(ValueError, msg=bad):
                runner.io_gate(row, "candidate")

    def test_determinism_cannot_silently_drop_logprobs(self):
        native = {"summary": "temperature-0 determinism", "identical": True, "logprobs": True,
                  "reference_text_sha256": "short", "long_text_sha256": "long"}
        runner.determinism_gate(json.dumps(native), logprobs=True)
        native["logprobs"] = False
        with self.assertRaises(ValueError):
            runner.determinism_gate(json.dumps(native), logprobs=True)

    def test_suite_cannot_pass_with_a_failed_step_changed_image_or_missing_cooling_rank(self):
        definition = {"name": "tiny", "steps": [{"id": "c1", "command": "run"}]}
        index = {"suite": definition, "suite_sha256": runner.suite.digest(definition), "finished_utc": "done",
                 "steps": [{"id": "c1", "command": "run", "exit": 0}], "metadata": {"identity": {
                  "live_problems": [], "client_dirty": False, "client_commit": "commit", "cluster_config": "config",
                  "nodes": {node: {"image_id": self.image["Id"], "checkout": "commit",
                            "image_layers": self.image["Layers"], "container_started_at": "boot"} for node in self.images}}}}
        conditions = {"halted": None, "fan_stop_failures": [],
                      "memory": {node: {"min_available_gib": 8, "swap_growth_gib": 0} for node in self.images},
                      "thermal": {node: {"thermal_slowdown_ms": 0} for node in self.images},
                      "cooling": [{"node": node, "final_c": 50} for node in self.images],
                      "fan_curve": [{"node": node} for node in self.images]}
        directory = self.root / "suite"
        def check():
            self.write(directory / "suite.json", index)
            self.write(directory / "conditions.json", conditions)
            return runner.native_suite_gate(directory, definition, "config", "commit", self.images)
        self.assertEqual(set(check()), set(self.images))
        index["steps"][0]["exit"] = 1
        with self.assertRaises(ValueError):
            check()
        index["steps"][0]["exit"] = 0
        index["metadata"]["identity"]["nodes"]["dgx1"]["image_id"] = "other"
        with self.assertRaises(ValueError):
            check()
        index["metadata"]["identity"]["nodes"]["dgx1"]["image_id"] = self.image["Id"]
        conditions["cooling"].pop()
        with self.assertRaises(ValueError):
            check()


if __name__ == "__main__":
    unittest.main()
