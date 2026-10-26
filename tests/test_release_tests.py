"""Release-test selection and the in-image runner's report parsing."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release_tests = load("release_tests", ROOT / "scripts/release_tests.py")
# The plugin imports pytest only for its hook decorator; the matching rule is plain Python.
sys.modules.setdefault("pytest", type(sys)("pytest"))
sys.modules["pytest"].hookimpl = getattr(sys.modules["pytest"], "hookimpl",
                                         lambda **kwargs: (lambda function: function))
offline_skip = load("offline_skip", ROOT / "scripts/release_tests/offline_skip.py")
runner = load("release_test_runner", ROOT / "scripts/release_tests/release_test_runner.py")


class SelectionTest(unittest.TestCase):
    def test_stack_paths(self):
        for path in ("tests/kernels/test_deepseek_v41_tilelang_rope.py",
                     "tests/v1/worker/test_b12x_roce_health.py",
                     "tests/v1/worker/test_startup_rendezvous.py"):
            self.assertTrue(release_tests.STACK.search(path), path)
        for path in ("tests/models/multimodal/processing/test_common.py",
                     "tests/v1/core/test_scheduler.py"):
            self.assertFalse(release_tests.STACK.search(path), path)


class ClassificationTest(unittest.TestCase):
    CONFIG = {
        "gate": {"tests/v1/worker/test_startup_rendezvous.py": "startup liveness"},
        "periodic": {"tests/kernels/test_deepseek_v41_engram_rows.py": "timing-based"},
        "excluded": {"tests/models/qwen4_exp/*": "a model this deployment does not serve",
                     "tests/v1/worker/test_startup_*": "overlaps the gate entry"},
    }

    def test_each_class_and_unclassified(self):
        classify = release_tests.classify
        self.assertEqual(classify("tests/kernels/test_deepseek_v41_engram_rows.py", self.CONFIG),
                         ["periodic"])
        self.assertEqual(classify("tests/models/qwen4_exp/test_b12x_qsa.py", self.CONFIG),
                         ["excluded"])
        self.assertEqual(classify("tests/v1/core/test_scheduler.py", self.CONFIG), [])
        # A gate file that also matches an exclusion is ambiguous, which check reports.
        self.assertEqual(classify("tests/v1/worker/test_startup_rendezvous.py", self.CONFIG),
                         ["gate", "excluded"])

    def test_config_entries_need_reasons_and_exact_gate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            for broken in ({"gate": {"tests/a.py": ""}, "periodic": {}, "excluded": {}},
                           {"gate": {"tests/*.py": "glob"}, "periodic": {}, "excluded": {}},
                           {"gate": {}, "periodic": {}},
                           {"gate": {"tests/a.py": "x"}, "periodic": {}, "excluded": {},
                            "deselect": {"tests/b.py::test_x": "not a gate file"}},
                           {"gate": {"tests/a.py": "x"}, "periodic": {}, "excluded": {},
                            "deselect": {"tests/a.py": "a whole file"}}):
                path.write_text(__import__("json").dumps(broken))
                with self.assertRaises(ValueError):
                    release_tests.load_config(path)


class BundleTest(unittest.TestCase):
    def test_o_direct_test_directories_are_on_the_node_cache(self):
        """The Engram rows' and the direct loader's tests read with O_DIRECT, which the
        image's overlay filesystem refuses: their files go under /c, the node's cache."""
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"files": []}')
            release_tests.bundle(str(manifest), "image:tag", str(Path(directory) / "bundle"), "label")
            env = __import__("json").loads((Path(directory) / "bundle" / "candidate.json").read_text())["env"]
        self.assertEqual(env["ENGRAM_ROWS_TEST_DIR"], "/c/engram-rows-test")
        self.assertEqual(env["DIRECT_IO_TEST_DIR"], "/c/direct-io-test")


class OfflineSkipTest(unittest.TestCase):
    def test_only_missing_resources_skip(self):
        missing = offline_skip.missing_resource
        self.assertEqual(missing("ValueError: 1 validation error for ModelConfig\n  Value error, "
                                 "Invalid repository ID or local directory specified: "
                                 "'facebook/opt-125m'."),
                         "Hugging Face facebook/opt-125m (offline)")
        self.assertEqual(missing("ModuleNotFoundError: No module named 'ray'"),
                         "ray (not in the serving image)")
        self.assertEqual(missing("LocalEntryNotFoundError: Cannot find an appropriate cached snapshot"),
                         "Hugging Face download (offline)")
        self.assertIsNone(missing("ValueError: 1 validation error for ModelConfig\n  Value error, "
                                  "max_model_len must be positive"))
        self.assertIsNone(missing("AssertionError: tensors differ"))


class RunnerReportTest(unittest.TestCase):
    def test_junit_outcomes_by_file(self):
        xml = """<testsuites><testsuite>
          <testcase classname="tests.test_a" file="tests/test_a.py" name="test_ok" time="0.5"/>
          <testcase classname="tests.test_a" file="tests/test_a.py" name="test_bad" time="1.0">
            <failure message="AssertionError: no&#10;more">trace</failure></testcase>
          <testcase classname="tests.test_b" name="test_skip" time="0"><skipped message="needs 2 GPUs"/></testcase>
          <testcase classname="tests.test_c" name="" time="0"><error message="collection failed"/></testcase>
        </testsuite></testsuites>"""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "r.xml"
            report.write_text(xml)
            found = runner.outcomes(report)
        self.assertEqual(sorted(found), ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"])
        summary = runner.summarize("tests/test_a.py", found["tests/test_a.py"], 1.5)
        self.assertEqual((summary["status"], summary["passed"], summary["failed"]), ("fail", 1, 1))
        self.assertEqual(summary["failures"][0]["message"], "AssertionError: no")
        self.assertEqual(runner.summarize("tests/test_b.py", found["tests/test_b.py"], 0)["status"], "ok")
        self.assertEqual(found["tests/test_c.py"][0]["outcome"], "error")


if __name__ == "__main__":
    unittest.main()
