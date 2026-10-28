"""Host-independent tests for the sweep writer (scripts/lab_sweep.py) and its stream workload."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


sweep_module = load("lab_sweep", "scripts/lab_sweep.py")
streams_module = load("distinct_streams", "scripts/distinct_streams.py")
lab = load("lab_for_sweep", "scripts/lab.py")
lab.spark.site_nodes = lambda: lab.spark.read_json("config/nodes.example.json")


SITE_SHAPE = mock.patch.object(lab.spark.cluster_recipe, "site_shape", return_value="tp3-triangle")


def setUpModule() -> None:
    # Planning resolves the recipe at the site's shape; the example map is a triangle.
    SITE_SHAPE.start()


def tearDownModule() -> None:
    # As in test_lab: pytest skips unittest's module cleanups, not tearDownModule.
    SITE_SHAPE.stop()

BASE_PATH = "config/cluster.json"
EXPERIMENT = "experiments/2026-10-06-prefetch-fill-rate"


class SweepTest(unittest.TestCase):
    def setUp(self) -> None:
        # The writer, not the recipe's current value: start from a base that leaves the variable unset.
        self.base = lab.spark.read_cluster(BASE_PATH, "tp4-ring")
        self.base["environment"].pop("VLLM_L2_PREFETCH_GRID", None)
        self.files = sweep_module.sweep(self.base, BASE_PATH, "VLLM_L2_PREFETCH_GRID", ["1", "2", "6"],
                                        EXPERIMENT, "sweep-tp4", [1, 4, 16])

    def test_each_arm_changes_only_the_variable(self) -> None:
        arms = {path: config for path, config in self.files.items() if "/sweep-grid" in path}
        self.assertEqual(sorted(arms), [f"{EXPERIMENT}/sweep-grid{v}.json" for v in ("1", "2", "6")])
        for path, config in arms.items():
            value = path.rsplit("grid", 1)[1].removesuffix(".json")
            self.assertEqual(config["environment"]["VLLM_L2_PREFETCH_GRID"], value)
            unchanged = json.loads(json.dumps(config))
            del unchanged["environment"]["VLLM_L2_PREFETCH_GRID"]
            self.assertEqual(unchanged, self.base)

    def test_spec_brackets_the_base_and_runs_every_stream_count(self) -> None:
        spec = self.files[f"{EXPERIMENT}/sweep-tp4.json"]
        (job,) = spec["jobs"]
        self.assertEqual((job["kind"], job["bracket"], job["profile"]), ("measure", True, "lean"))
        self.assertEqual(job["arms"][0], {"config": BASE_PATH, "label": "base"})
        self.assertEqual([arm["label"] for arm in job["arms"][1:]], ["grid1", "grid2", "grid6"])
        self.assertEqual([extra[2] for extra in job["extras"]], ["c1-distinct", "c4-distinct", "c16-distinct"])
        steps = lab.plan(spec)
        boots = [step["config"] for step in steps if step["kind"] == "boot"]
        self.assertEqual(boots, [BASE_PATH, *[f"{EXPERIMENT}/sweep-grid{v}.json" for v in ("1", "2", "6")],
                                 BASE_PATH])
        scripts = {step["argv"][0] for step in steps if step["kind"] == "script"}
        self.assertEqual(scripts, {"scripts/distinct_streams.py"})

    def test_the_base_setting_is_not_measured_twice(self) -> None:
        base = json.loads(json.dumps(self.base))
        base["environment"]["VLLM_L2_PREFETCH_GRID"] = "4"
        files = sweep_module.sweep(base, BASE_PATH, "VLLM_L2_PREFETCH_GRID", ["2", "4"], EXPERIMENT, "r", [1])
        self.assertNotIn(f"{EXPERIMENT}/sweep-grid4.json", files)
        with self.assertRaises(ValueError):
            sweep_module.sweep(base, BASE_PATH, "VLLM_L2_PREFETCH_GRID", ["4"], EXPERIMENT, "r", [1])

    def test_bad_values_are_refused(self) -> None:
        for values in ([], ["2", "2"], ["2 3"], ["../x"]):
            with self.assertRaises(ValueError):
                sweep_module.sweep(self.base, BASE_PATH, "VLLM_L2_PREFETCH_GRID", values, EXPERIMENT, "r", [1])
        with self.assertRaises(ValueError):
            sweep_module.sweep(self.base, BASE_PATH, "VLLM_L2_PREFETCH_GRID", ["2"], EXPERIMENT, "r", [0])

    def test_sweeps_are_written_under_experiments_only(self) -> None:
        sweep_module.check_destination(EXPERIMENT)
        for place in ("config", "experiments", "results/private/lab/x", "experiments/../config"):
            with self.assertRaises(ValueError):
                sweep_module.check_destination(place)


class DistinctStreamsTest(unittest.TestCase):
    def test_summary_is_the_line_lab_tables_read(self) -> None:
        line = streams_module.summary(4, [100.0, 102.0, 101.0])
        self.assertEqual(line["summary"], "c4 distinct prompts")
        self.assertEqual((line["tps_mean"], line["samples"]), (101.0, 3))
        self.assertGreater(line["ci95_pct"], 0)

    def test_prompts_are_distinct(self) -> None:
        self.assertEqual(len(set(streams_module.PROMPTS)), len(streams_module.PROMPTS))
        self.assertGreaterEqual(len(streams_module.PROMPTS), 16)

    def test_requests_beyond_the_samples_own_are_foreign(self) -> None:
        """A sample counts the server's finished requests: its own streams are expected,
        anything more was someone else's load on the server (2026-10-18: a production
        client on a lab arm made a gate read 3-4x slow)."""
        def fake_stream(base, model, prompt, tokens, out):
            out.update(start=0.0, first=1.0, last=3.0, tokens=tokens, offset=0.0)

        for counts, foreign in (([10.0, 14.0], 0), ([10.0, 21.0], 7), ([None, 14.0], None)):
            with mock.patch.object(streams_module, "stream", fake_stream), \
                    mock.patch.object(streams_module, "finished_requests", side_effect=counts):
                result = streams_module.sample("http://x", "m", streams_module.PROMPTS[:4], 16)
            self.assertEqual(result["foreign_requests"], foreign)


if __name__ == "__main__":
    unittest.main()
