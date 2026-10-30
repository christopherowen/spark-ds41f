"""Each release's frozen recipe is what it qualified, and production is the newest release.

A promotion adds ``releases/<name>/`` (the recipe and its lock, never edited afterwards) and a
baseline manifest naming them, the arm each shape qualified on, and any change made to the
recipe after the release. No test names a release: these checks read the manifests."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import recipe  # noqa: E402
from test_recipe import RENDERED, host_mounts, load_spark  # noqa: E402

# What names a configuration rather than what it serves.
NAMING = ("promoted_baseline", "upstreams_config", "benchmark_reference", "shape", "notes")


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def releases() -> dict[str, dict]:
    """The baseline manifests that freeze their recipe, by manifest name."""
    out = {}
    for path in sorted((ROOT / "manifests/baselines").glob("*.json")):
        manifest = json.loads(path.read_text())
        if "recipe" in manifest:
            out[path.stem] = manifest
    return out


def serving(cluster: dict) -> dict:
    out = {k: v for k, v in host_mounts(cluster).items() if k not in NAMING}
    out["environment"] = {k: v for k, v in out["environment"].items() if k not in RENDERED}
    return out


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.releases = releases()
        self.spark = load_spark()

    def test_each_release_is_its_qualified_arms(self):
        """At each shape it qualified, a release's recipe resolves to the arm it was measured
        on, apart from the names, the fabric settings sparknet renders from the node map, the
        release tag of the same image, and the changes its manifest records after release."""
        self.assertTrue(self.releases)
        for name, manifest in self.releases.items():
            frozen = load(manifest["recipe"])
            self.assertEqual(frozen["promoted_baseline"], name)
            self.assertEqual(frozen["container"]["image"], manifest["image_tag"], name)
            self.assertTrue(manifest["qualified_arms"], name)
            for shape, arm_path in manifest["qualified_arms"].items():
                with self.subTest(release=name, shape=shape):
                    arm = self.spark.read_cluster(arm_path)
                    if arm["container"]["image"] != manifest["image_tag"]:
                        self.assertEqual(arm["container"]["image"], manifest["build"]["built_as"])
                        arm["container"]["image"] = manifest["image_tag"]
                    for change in manifest.get("changed_after_release", []):
                        self.assertTrue(change["why"])
                        if shape in change["shapes"]:
                            arm = recipe.overlay({"extends": {}, **change["overlay"]}, arm)
                    self.assertEqual(serving(recipe.resolve(frozen, shape)), serving(arm))

    def test_each_release_is_frozen_with_its_lock(self):
        for name, manifest in self.releases.items():
            directory = Path(manifest["recipe"]).parent
            frozen = load(manifest["recipe"])
            self.assertEqual(frozen["upstreams_config"], f"{directory}/upstreams.lock.json", name)
            self.assertEqual(load(frozen["upstreams_config"])["generated_from_baseline"], name)
            references = {entry.get("benchmark_reference") for entry in frozen["shapes"].values()}
            self.assertIn(manifest["benchmark_reference"], references, name)

    def test_production_is_the_newest_release(self):
        """config/cluster.json and upstreams.lock.json are the newest release's copies; a
        recipe change waits in a candidate's arm until it is released."""
        production = load("config/cluster.json")
        manifest = self.releases[production["promoted_baseline"]]
        self.assertEqual(max(self.releases, key=lambda n: self.releases[n]["captured_at"]),
                         production["promoted_baseline"])
        frozen = load(manifest["recipe"])
        self.assertEqual({k: v for k, v in production.items() if k != "upstreams_config"},
                         {k: v for k, v in frozen.items() if k != "upstreams_config"})
        self.assertNotIn("upstreams_config", production)
        self.assertEqual(load("upstreams.lock.json"), load(frozen["upstreams_config"]))

    def test_each_release_follows_the_one_it_names(self):
        for name, manifest in self.releases.items():
            previous = manifest.get("previous_baseline")
            if previous in self.releases:
                self.assertLess(self.releases[previous]["captured_at"], manifest["captured_at"], name)


if __name__ == "__main__":
    unittest.main()
