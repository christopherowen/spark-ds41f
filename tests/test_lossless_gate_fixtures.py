"""Test fixtures must not replace runtime code or shrink the selected catalog."""

import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"


class GateFixturesTests(unittest.TestCase):
    def test_only_pinned_test_sources_are_overridden_on_the_same_runtime_image(self):
        manifest = json.loads((HERE / "gate-fixtures/manifest.json").read_text())
        patch = HERE / "gate-fixtures/fixes.patch"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), manifest["patch_sha256"])
        expected = {row["path"]: row["fixture_sha256"] for row in manifest["original_files"] if row["fixture_sha256"]}
        for arm in ("expert-scales", "engram"):
            old, new = (HERE / arm / name for name in ("full-gate", "full-gate-fixtures"))
            a, b = (json.loads((path / "candidate.json").read_text()) for path in (old, new))
            self.assertEqual(a["image"], b["image"])
            self.assertEqual(a["env"], b["env"])
            self.assertNotEqual(a["argv"][-1], b["argv"][-1])
            self.assertEqual((old / "manifest.json").read_bytes(), (new / "manifest.json").read_bytes())
            added = {tuple(row) for row in b["mounts"]} - {tuple(row) for row in a["mounts"]}
            self.assertEqual(len(added), len(expected))
            for source, target in added:
                relative = target.removeprefix("/opt/spark-ds41f/candidate/vllm/")
                self.assertIn(relative, expected)
                self.assertTrue(relative.startswith("tests/v1/worker/"))
                self.assertEqual(hashlib.sha256((new / source).read_bytes()).hexdigest(), expected[relative])
            original = json.loads((HERE / arm / "full-gate-original.json").read_text())
            current = json.loads((HERE / arm / "full-gate.json").read_text())
            self.assertNotEqual(original["run"], current["run"])
            self.assertNotEqual(original["jobs"][0]["bundles"][0]["bundle"],
                                current["jobs"][0]["bundles"][0]["bundle"])


if __name__ == "__main__":
    unittest.main()
