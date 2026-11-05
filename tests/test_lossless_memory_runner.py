"""Failure-path checks for owned cluster-window restoration, with no node calls."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"
with patch.object(sys, "path", [str(EXPERIMENT), *sys.path]):
    spec = importlib.util.spec_from_file_location("lossless_runner_tests", EXPERIMENT / "run_fp8.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.here = self.root / "experiments/memory"
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.hold = {"window": "owned", "holder": runner.HOLDER, "production_config": runner.RESTORE,
                     "since": "2026-11-04T00:00:00Z"}
        self.closed, self.jobs = [], []
        self.pending, self.change_to_foreign, self.omit_verdict = False, False, False
        self.closure_reason = "cluster requested (fixture)"
        self.request_path = self.root / "spark-request.json"
        self.management_unreachable = []
        self.restore_refused = False
        self.write(self.here / "base.json", {"image_id": "sha256:" + "a" * 64, "model_revision": "b" * 40})
        self.write(self.here / "fp8/kernel-32.json",
                   {"run": "first", "jobs": [{"bundles": [{"bundle": "experiments/memory/fp8/bundle", "node": "dgx1"}]}]})
        self.write(self.here / "fp8/bundle/candidate.json", {"argv": ["probe.py", "--out", "/c/native.json"]})
        self.write(self.cache / "native.json", {"gpu_byte_exact": False})

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def close(self, reason):
        self.closed.append(reason)
        self.hold = None
        return True

    def run_command(self, argv, **kwargs):
        if "bin/spark" in argv:
            self.assertNotIn("--apply", argv)
            return SimpleNamespace(returncode=0, stdout="live: running: 3 nodes, TP3\nplan only\n" +
                                   ("--apply would refuse" if self.restore_refused and "start" in argv else ""), stderr="")
        raise AssertionError("unexpected process: " + repr(argv))

    def kernel(self, step):
        self.jobs.append(step)
        if not self.omit_verdict:
            self.write(self.root / "results/private/lab/first/verdicts.json",
                       {"agree": True, "verdicts": [{"passed": True, "exit": 0,
                        "image": "sha256:" + "a" * 64, "errors": [], "failed": []}]})
        if self.change_to_foreign:
            self.hold = {"window": "foreign", "holder": "spark-lab: another owner",
                         "production_config": runner.RESTORE}
        return []

    def invoke(self, *, format="huffman", lanes=32, chunk=None):
        fake_lab = SimpleNamespace(
            read_hold=lambda: self.hold,
            window_should_close=lambda hold: self.closure_reason if self.pending else None,
            REQUEST=self.request_path,
            HOLD=self.root / "hold.json",
            now=lambda: "2026-11-04T00:00:00Z", beat=lambda note: None,
            write_json_atomic=self.write, compile_cache=lambda candidate: self.cache,
            window_close=self.close, production_live=lambda: True,
            stop_cluster=lambda: True, run_kernel_bundles=self.kernel,
            plan=lambda spec: [{"kind": "kernel"}])
        process = SimpleNamespace(
            check_output=lambda argv, **kw: b"" if "status" in argv else "c" * 40,
            run=self.run_command, STDOUT=runner.subprocess.STDOUT)
        argv = ["run_fp8.py", "--window", "owned", "--format", format, "--lanes", str(lanes),
                "--out", "results/private/round", "--apply"]
        if chunk is not None:
            argv += ["--chunk-bytes", str(chunk)]
        with patch.multiple(runner, ROOT=self.root, HERE=self.here, lab=fake_lab,
                            subprocess=process, os=SimpleNamespace(uname=lambda: SimpleNamespace(nodename="dgx1"))), \
                patch.object(runner, "probe_management", return_value=self.management_unreachable), \
                patch.object(sys, "argv", argv):
            return runner.main()

    def receipt(self):
        return json.loads((self.root / "results/private/round/receipt.json").read_text())

    def test_failed_native_gate_stops_before_repeat_and_restores_owned_window(self):
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(len(self.jobs), 1)
        self.assertEqual(len(self.closed), 1)
        receipt = self.receipt()
        self.assertTrue(receipt["restored"])
        self.assertNotIn("completed", receipt)
        self.assertIn("inexact decoder receipt", receipt["error"])

    def test_failed_plane_lz4_gate_stops_the_matrix_and_restores_owned_window(self):
        spec = json.loads((self.here / "fp8/kernel-32.json").read_text())
        self.write(self.here / "fp8/plane-lz4/kernel-512-128.json", spec)
        self.assertEqual(self.invoke(format="plane-lz4", lanes=128, chunk=512), 1)
        self.assertEqual(len(self.jobs), 1)
        self.assertEqual(len(self.closed), 1)
        self.assertTrue(self.receipt()["restored"])
        self.assertNotIn("completed", self.receipt())

    def test_missing_native_verdict_restores_owned_window(self):
        self.omit_verdict = True
        self.assertEqual(self.invoke(), 1)
        self.assertTrue(self.receipt()["restored"])
        self.assertEqual(len(self.closed), 1)
        self.assertIn("FileNotFoundError", self.receipt()["error"])

    def test_new_foreign_window_is_never_closed_during_failure_cleanup(self):
        self.change_to_foreign = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.closed, [])
        self.assertEqual(self.hold["window"], "foreign")
        self.assertIn("restore_deferred", self.receipt())

    def test_pending_shared_request_runs_no_job_and_releases_own_window(self):
        self.pending = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.jobs, [])
        self.assertEqual(len(self.closed), 1)
        self.assertTrue(self.receipt()["restored"])

    def test_existing_foreign_holder_is_refused_before_any_job_or_cleanup(self):
        self.hold["holder"] = "spark-lab: another owner"
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.closed, [])
        self.assertFalse((self.root / "results/private/round").exists())

    def test_return_queued_after_opening_stays_visible_until_current_screen_finishes(self):
        self.pending = True
        self.write(self.request_path, {"after_window": "owned", "since": "2026-11-04T00:00:00.1+00:00"})
        self.assertEqual(self.invoke(), 1)  # fixture's byte gate fails
        self.assertEqual(len(self.jobs), 1)
        self.assertTrue(self.request_path.exists())
        self.assertIn("queued_return_request", self.receipt())
        self.assertTrue(self.receipt()["restored"])

    def test_time_cap_is_not_treated_as_a_queued_return_request(self):
        self.pending = True
        self.closure_reason = "time cap reached"
        self.write(self.request_path, {"after_window": "owned", "since": "2026-11-04T00:00:00.1+00:00"})
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.jobs, [])
        self.assertTrue(self.receipt()["restored"])

    def test_fixed_field_plan_uses_distinct_first_repeat_bundles(self):
        argv = ["run_fp8.py", "--window", "plan", "--format", "fixed-fields",
                "--out", "results/private/fixed-field-plan"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(), 0)
        specs = json.loads(output.getvalue())["specs"]
        self.assertEqual(len(set(specs)), 6)
        self.assertTrue(all("/fp8/fixed/" in path for path in specs))
        self.assertTrue(any("512-repeat" in path for path in specs))

    def test_plane_lz4_plan_keeps_all_eight_observations_and_sources_distinct(self):
        argv = ["run_fp8.py", "--window", "plan", "--format", "plane-lz4",
                "--out", "results/private/plane-plan"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(), 0)
        paths = json.loads(output.getvalue())["specs"]
        self.assertEqual(len(set(paths)), 8)
        runs, outputs = set(), set()
        for path in paths:
            spec = json.loads((runner.ROOT / path).read_text())
            runs.add(spec["run"])
            bundle = runner.ROOT / spec["jobs"][0]["bundles"][0]["bundle"]
            candidate = json.loads((bundle / "candidate.json").read_text())
            outputs.add(candidate["argv"][-1])
            self.assertIn("--chunk-bytes", candidate["argv"])
            for source, _ in candidate["mounts"]:
                if not source.startswith("/"):
                    self.assertTrue((runner.ROOT / source).is_file(), source)
        self.assertEqual((len(runs), len(outputs)), (8, 8))

    def test_management_loss_never_launches_restoration(self):
        self.management_unreachable = ["dgx2"]
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.closed, [])
        hold = json.loads((self.root / "hold.json").read_text())
        self.assertEqual(hold["holder"], "lossless-memory-hardware-review")
        self.assertFalse(self.receipt()["restored"])

    def test_refused_restoration_plan_is_retained_without_another_launch(self):
        self.restore_refused = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.closed, [])
        hold = json.loads((self.root / "hold.json").read_text())
        self.assertEqual(hold["holder"], "lossless-memory-restoration-review")


if __name__ == "__main__":
    unittest.main()
