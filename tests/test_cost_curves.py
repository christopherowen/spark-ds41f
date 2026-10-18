"""DSpark cost curves per build: a profile without a pinned directory gets its build's."""

import copy
import importlib.machinery
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)

EXAMPLE_NODES = {"3": "config/nodes.example.json", "4": "config/examples/nodes-ring4.json"}


def head(cluster: dict) -> dict:
    nodes = spark.read_json(EXAMPLE_NODES[spark.serve_arg(cluster, "--tensor-parallel-size")])
    return next(node for node in nodes["nodes"] if node.get("head"))


class CostCurveDirectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tp3 = spark.read_cluster("config/cluster.json", "tp3-triangle")
        self.tp4 = spark.read_cluster("config/cluster.json", "tp4-ring")

    def directory(self, cluster: dict) -> str:
        return spark.expected_environment(cluster, head(cluster))["SPARK3_DSPARK_COST_DIR"]

    def test_every_shape_derives_its_build_directory(self) -> None:
        # Since r6v the ring, like the triangle, schedules on its build's own curves.
        for cluster in (self.tp3, self.tp4):
            derived = self.directory(cluster)
            self.assertRegex(derived, r"^/cache/kkref/dspark-costs/build-[0-9a-f]{16}$")
            self.assertEqual(derived, f"{spark.DSPARK_COST_ROOT}/build-{spark.build_fingerprint(cluster)}")
            self.assertNotIn("SPARK3_DSPARK_COST_DIR", cluster["environment"])
        # The frozen r6n arm on the ring keeps the curves r6n served it with (r6h's).
        r6n = spark.read_cluster("experiments/2026-10-17-r6u/r6n-tp4-dev-1017-ring-curves.json")
        self.assertEqual(self.directory(r6n), "/cache/kkref/dspark-costs/ring4-r6h-headout-20261012")

    def test_same_build_same_curves(self) -> None:
        # A setting that is not code (the dev-mode RPC) shares the build's curves.
        dev = copy.deepcopy(self.tp3)
        dev["environment"]["VLLM_SERVER_DEV_MODE"] = "1"
        self.assertEqual(self.directory(dev), self.directory(self.tp3))
        # The ring's release arm (dev mode) and the recipe are the same build, and so are the
        # triangle's r6x arm and the recipe. (r6x reads no curves; the directory still names
        # the build.)
        arm = spark.read_cluster("experiments/2026-10-18-r6x/r6x-tp4-dev-1017.json")
        self.assertEqual(self.directory(arm), self.directory(self.tp4))
        arm = spark.read_cluster("experiments/2026-10-18-r6x/r6x-tp3-1017.json")
        self.assertEqual(self.directory(arm), self.directory(self.tp3))
        # r6v's arms keep the curves its production ran on, so it stays a fixed reference.
        r6v = spark.read_cluster("experiments/2026-10-17-r6v/r6v-tp4-dev-1017.json")
        self.assertEqual(self.directory(r6v), "/cache/kkref/dspark-costs/build-2a95f7780992adb4")
        r6v = spark.read_cluster("experiments/2026-10-17-r6v/r6v-tp3-1017.json")
        self.assertEqual(self.directory(r6v), "/cache/kkref/dspark-costs/build-452635dc54559431")
        # r6u's TP3 release arm keeps r6u's curves, the ones its production ran on.
        r6u = spark.read_cluster("experiments/2026-10-17-r6u/r6u-tp3-dev-1017.json")
        self.assertEqual(self.directory(r6u), "/cache/kkref/dspark-costs/build-111f0abb94a12639")
        # The frozen r6s recipe keeps r6s's TP3 curves, the ones its production ran on.
        r6s = spark.read_cluster("experiments/2026-10-16-r6s/r6s-tp3-dev.json")
        self.assertEqual(self.directory(r6s), "/cache/kkref/dspark-costs/build-75cd5b96c4ec4942")
        # The frozen r6n recipe keeps r6n's TP3 curves, the ones its production ran on.
        r6n = spark.read_cluster("experiments/2026-10-15-r6n-tp3/r6n-tp3.json")
        self.assertEqual(self.directory(r6n), "/cache/kkref/dspark-costs/build-77b3fa3c35f1ce6d")
        # A traced arm times differently: its curves never become the build's.
        traced = spark.read_cluster("experiments/2026-10-15-r6n-tp3/r6n-tp3-traced.json")
        self.assertNotEqual(self.directory(traced), self.directory(self.tp3))

    def test_another_build_gets_its_own(self) -> None:
        other = copy.deepcopy(self.tp3)
        other["container"]["expected_labels"]["local.spark3.vllm.tree"] = "0" * 40
        self.assertNotEqual(self.directory(other), self.directory(self.tp3))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            overlay = Path(tmp) / "kernel.py"
            overlay.write_text("one")
            mounted = copy.deepcopy(self.tp3)
            source = "{home}/projects/spark-ds41f/" + str(overlay.relative_to(ROOT))
            mounted["container"]["mounts"].append([source, "/opt/spark3/candidate/vllm/kernel.py", "ro"])
            first = self.directory(mounted)
            self.assertNotEqual(first, self.directory(self.tp3))
            overlay.write_text("two")  # the same path with other code
            self.assertNotEqual(self.directory(mounted), first)

    def test_a_pinned_directory_wins(self) -> None:
        r6h = spark.read_cluster("experiments/2026-10-14-one-recipe/tp3-r6h-production.json")
        self.assertEqual(self.directory(r6h), "/cache/kkref/dspark-costs/tp3-r6h-20261012")


if __name__ == "__main__":
    unittest.main()
