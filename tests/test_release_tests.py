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
runner = load("release_test_runner", ROOT / "scripts/release_tests/release_test_runner.py")


class SelectionTest(unittest.TestCase):
    def test_imports_name_modules_and_imported_names(self):
        text = ("import vllm.envs as envs\n"
                "from vllm.v1.worker import gpu_worker, utils\n"
                "from vllm.models.deepseek_v4_1.tilelang import (\n    prepare,\n    vocab,\n)\n"
                "def f():\n    from vllm.config import VllmConfig\n")
        self.assertEqual(release_tests.imported_modules(text), {
            "vllm.envs", "vllm.v1.worker", "vllm.v1.worker.gpu_worker", "vllm.v1.worker.utils",
            "vllm.models.deepseek_v4_1.tilelang", "vllm.models.deepseek_v4_1.tilelang.prepare",
            "vllm.models.deepseek_v4_1.tilelang.vocab", "vllm.config", "vllm.config.VllmConfig"})

    def test_stack_paths(self):
        for path in ("tests/kernels/test_deepseek_v41_tilelang_rope.py",
                     "tests/v1/worker/test_b12x_roce_health.py",
                     "tests/v1/worker/test_startup_rendezvous.py"):
            self.assertTrue(release_tests.STACK.search(path), path)
        for path in ("tests/models/multimodal/processing/test_common.py",
                     "tests/v1/core/test_scheduler.py"):
            self.assertFalse(release_tests.STACK.search(path), path)


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
