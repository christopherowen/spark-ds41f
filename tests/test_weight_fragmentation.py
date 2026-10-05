"""Read-only fragmentation checks and absolute (not peer-relative) findings."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


probe = load("weight_fragmentation", ROOT / "scripts/weight_fragmentation.py")
spark3 = load("spark3", ROOT / "bin/spark3")


class FragmentationTest(unittest.TestCase):
    def test_score_zero_does_not_mean_ideal(self):
        text = "Total/best extents\t37/1\nFragmentation score\t0\n"
        self.assertEqual(probe.extent_counts(text, 1), [(37, 1)])

    def test_incomplete_or_invalid_maps_are_not_clean(self):
        for text in ("", "Total/best extents 1/0\n", "Total/best extents 1/2\n"):
            with self.assertRaises(ValueError):
                probe.extent_counts(text, 1)
        with self.assertRaises(ValueError):
            probe.extent_counts("Total/best extents 1/1\n", 2)

    def report(self, current, best):
        return {"shards": [{"name": "shard", "path": "/weights/a blob", "current": current, "best": best}]}

    def test_ideal_large_files_remain_quiet(self):
        self.assertEqual(spark3.report_weight_fragmentation({"dgx1": self.report(48, 48)}), [])

    def test_every_nonideal_file_warns_even_when_nodes_match(self):
        reports = {node: self.report(49, 48) for node in ("dgx1", "dgx2", "dgx3")}
        findings = spark3.report_weight_fragmentation(reports)
        self.assertEqual(len(findings), 4)
        self.assertTrue(all(isinstance(f, spark3.Warn) for f in findings))
        self.assertIn("49 current / 48 best", findings[0])

    def test_unknown_is_not_reported_as_clean(self):
        findings = spark3.report_weight_fragmentation({"dgx3": {"error": "sudo unavailable"}})
        self.assertIn("cannot check", findings[0])

    def test_correction_commands_are_quoted_and_only_for_nonideal_files(self):
        findings = spark3.report_weight_fragmentation({
            "dgx1": self.report(48, 48), "dgx3": self.report(49, 48),
        }, commands=True)
        self.assertEqual(findings[-1], "sudo e4defrag -v '/weights/a blob'")
        self.assertNotIn("dgx1", "\n".join(findings))

    def test_probe_resolves_indexed_symlinks_and_always_checks_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blob"
            blob.write_bytes(b"test")
            (root / "shard.safetensors").symlink_to(blob)
            (root / "unindexed.safetensors").write_bytes(b"ignored")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "shard.safetensors", "b": "shard.safetensors"},
            }))
            result = subprocess.CompletedProcess([], 0, "Total/best extents 2/1\n", "")
            with mock.patch.object(probe.subprocess, "run", return_value=result) as run:
                report = probe.measure(root)
            command = run.call_args.args[0]
            self.assertEqual(command[-3:], ["e4defrag", "-c", str(blob.resolve())])
            self.assertNotIn("-v", command)
            self.assertEqual(len(report["shards"]), 1)
            self.assertEqual(report["shards"][0]["current"], 2)


if __name__ == "__main__":
    unittest.main()
