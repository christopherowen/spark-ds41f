"""Release-test selection and the in-image runner's report parsing."""
import importlib.util
import json
import os
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
# deepseek_models installs its hub cache when imported; a missing checkpoint leaves this
# process's environment alone, so the tests call its functions directly.
_named = os.environ.get("DS41F_SNAPSHOT")
os.environ["DS41F_SNAPSHOT"] = "/nonexistent/ds41f-snapshot"
deepseek_models = load("deepseek_models", ROOT / "scripts/release_tests/deepseek_models.py")
if _named is None:
    del os.environ["DS41F_SNAPSHOT"]
else:
    os.environ["DS41F_SNAPSHOT"] = _named


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

    def test_served_checkpoint_is_mounted_for_the_deepseek_plugin(self):
        """The bundle mounts the recipe's served checkpoint read-only, as serving does, and
        names its snapshot for deepseek_models.py, which the runner loads."""
        recipe = json.loads((ROOT / "config/cluster.json").read_text())
        profile = recipe["models"][recipe["model"]]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"files": []}')
            release_tests.bundle(str(manifest), "image:tag", str(Path(directory) / "bundle"), "label")
            bundle = Path(directory) / "bundle"
            candidate = json.loads((bundle / "candidate.json").read_text())
            self.assertTrue((bundle / "deepseek_models.py").is_file())
        mounts = dict((destination, source) for source, destination in candidate["mounts"])
        self.assertEqual(mounts["/b/deepseek_models.py"], "deepseek_models.py")
        self.assertEqual(mounts["/hub"], profile["hub"].replace("{home}", "~"))
        self.assertEqual(candidate["env"]["DS41F_SNAPSHOT"], f"/hub/{profile['snapshot']}")
        self.assertIn("deepseek_models", runner.pytest.__code__.co_consts)


class DeepseekModelsTest(unittest.TestCase):
    def snapshot(self, root: Path) -> Path:
        """A hub entry like the served one: snapshot files linking into blobs."""
        repo = root / "models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
        snapshot = repo / "snapshots" / "1bae0e10"
        (repo / "blobs").mkdir(parents=True)
        snapshot.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors.index.json", "model-00001-of-00048.safetensors"):
            (repo / "blobs" / name).write_text(name)
            (snapshot / name).symlink_to(Path("../../blobs") / name)
        return snapshot

    def test_stand_ins_get_config_and_tokenizer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.snapshot(Path(directory) / "hub")
            cache = Path(directory) / "overlay"
            deepseek_models.build_cache(snapshot, cache)
            for repo in deepseek_models.STAND_INS:
                root = deepseek_models.repo_dir(cache, repo)
                self.assertEqual((root / "refs" / "main").read_text(), "1bae0e10")
                files = sorted(p.name for p in (root / "snapshots" / "1bae0e10").iterdir())
                self.assertEqual(files, sorted(deepseek_models.STAND_IN_FILES), repo)
                self.assertEqual((root / "snapshots" / "1bae0e10" / "tokenizer.json").read_text(),
                                 "tokenizer.json")
            served = deepseek_models.repo_dir(cache, deepseek_models.REPO) / "snapshots" / "1bae0e10"
            self.assertEqual(len(list(served.iterdir())), 5)
            self.assertEqual((served / "model-00001-of-00048.safetensors").read_text(),
                             "model-00001-of-00048.safetensors")

    def test_only_deepseek_config_refusals_skip(self):
        refusal = deepseek_models.refusal
        self.assertIn("aligns prefill chunks", refusal(
            "ValueError: aligned prefill chunks need long_prefill_token_threshold (0) plus 16 decode "
            "rows within the 8192-token step budget"))
        self.assertIn("DBO", refusal("ValidationError: Value error, DeepSeek V4.1 Engram does not "
                                     "support DBO or microbatching."))
        self.assertIn("DSpark", refusal("ValidationError: DeepSeek V4.1 has no classic-MTP draft: ..."))
        self.assertIn("tokenizer", refusal(
            "ValueError: DeepseekV4VLMultiModalProcessor requires a tokenizer and cannot be "
            "initialized when `skip_tokenizer_init=True`."))
        # A threshold the test did set, and any other failure, stay failures.
        self.assertIsNone(refusal("ValueError: aligned prefill chunks need long_prefill_token_threshold "
                                  "(400) plus 16 decode rows within the 64-token step budget"))
        self.assertIsNone(refusal("AssertionError: assert 2 == 3"))

    def test_snapshot_is_found_in_the_hub_cache_or_named(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.snapshot(Path(directory) / "hub")
            saved = {name: os.environ.pop(name, None) for name in ("DS41F_SNAPSHOT", "HF_HUB_CACHE")}
            try:
                os.environ["HF_HUB_CACHE"] = str(Path(directory) / "hub")
                self.assertEqual(deepseek_models.find_snapshot(), snapshot)
                os.environ["DS41F_SNAPSHOT"] = str(Path(directory) / "missing")
                self.assertIsNone(deepseek_models.find_snapshot())
                os.environ["DS41F_SNAPSHOT"] = str(snapshot)
                self.assertEqual(deepseek_models.find_snapshot(), snapshot)
            finally:
                for name, value in saved.items():
                    os.environ.pop(name, None)
                    if value is not None:
                        os.environ[name] = value


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
        self.assertEqual(missing("ValidationError: 1 validation error for ParallelConfig\n  Value error, "
                                 "World size (4) is larger than the number of available GPUs (1) in this node."),
                         "4 GPUs (the node has 1)")


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
