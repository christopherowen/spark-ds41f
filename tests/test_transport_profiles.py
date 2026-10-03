"""Named profiles must preserve measured settings and reject silent drift."""

import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3_tuning_test", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
profiles = spark3.transport_profiles


class TransportProfilesTest(unittest.TestCase):
    def resolve(self, name):
        return profiles.resolve(ROOT, profiles.DEFAULT_PROFILES, name)

    def test_profiles_reproduce_base_execution_settings(self):
        for name, node_file in (("tp3", "config/nodes.example.json"),
                                ("tp4", "config/examples/nodes-ring4.json")):
            with self.subTest(name=name):
                profile, resolved = self.resolve(name)
                base = spark3.read_json(profile["base_config"])
                self.assertEqual(resolved, base)
                nodes = spark3.read_json(node_file)
                result = profiles.materialize(profile, resolved, nodes, node_file)
                self.assertEqual(result["environment"], base["environment"])
                self.assertEqual(result["serve_args"], base["serve_args"])
                self.assertEqual(result["container"], base["container"])
                self.assertEqual(spark3.topology.problems(result, nodes), [])
                self.assertFalse(result["deployment"]["launch_enabled"])
                self.assertNotIn("branch", result["deployment"])
                self.assertEqual(resolved, base)

    def test_wrong_topology_rejected(self):
        profile, base = self.resolve("tp3")
        with self.assertRaisesRegex(ValueError, "does not match"):
            profiles.materialize(profile, base, spark3.read_json("config/examples/nodes-ring4.json"), "nodes.json")

    def test_catalog_validation(self):
        original = spark3.read_json(profiles.DEFAULT_PROFILES)
        mutations = [
            ("tp4", lambda p: p.update(base_sha256="0" * 64), "base config changed"),
            ("tp4", lambda p: p["rocenante"].update(B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES="4194304"), "dispatch"),
            ("tp4", lambda p: p["rocenante"].update(B12X_ROCE_STREAM_CHUNK_BYTES="65536"), "exactly"),
            ("tp4", lambda p: p["nccl"].update(NCCL_BUFFSIZE="-1"), "positive integer"),
            ("tp4", lambda p: p["nccl"].update(NCCL_BUFFSIZE=None), "explicit values"),
            ("tp4", lambda p: p["nccl"].update(NCCL_THREAD_THRESHOLDS="1 2"), "six"),
            ("tp4", lambda p: p["nccl"].update(NCCL_SWITCHLESS_BIDIRECTIONAL="0"), "balanced TP4"),
            ("tp3", lambda p: p["rocenante"].update(B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES="1048576"), "does not provide"),
            ("tp4", lambda p: p.update(base_config="../elsewhere.json"), "relative"),
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "profiles.json"
            for name, mutate, message in mutations:
                with self.subTest(message=message):
                    catalog = copy.deepcopy(original)
                    mutate(catalog["profiles"][name])
                    path.write_text(json.dumps(catalog))
                    with self.assertRaisesRegex(ValueError, message):
                        profiles.resolve(ROOT, str(path.relative_to(ROOT)), name)

    def test_edited_channels_are_validated_against_ring_policy(self):
        profile, resolved = self.resolve("tp4")
        resolved["environment"]["NCCL_MIN_NCHANNELS"] = "2"
        with self.assertRaisesRegex(ValueError, "matching"):
            profiles.materialize(profile, resolved, spark3.read_json("config/examples/nodes-ring4.json"), "nodes.json")

    def test_cli_materializes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as directory:
            output = str((Path(directory) / "candidate.json").relative_to(ROOT))
            args = spark3.parser().parse_args([
                "tuning", "create", "tp4", "--nodes-config", "config/examples/nodes-ring4.json", "--output", output])
            with mock.patch("builtins.print"):
                self.assertEqual(args.func(args), 0)
            generated = spark3.read_json(output)
            self.assertEqual(generated["tuning_origin"]["profile"], "tp4")
            self.assertFalse(generated["deployment"]["launch_enabled"])
            with self.assertRaises(SystemExit):
                args.func(args)

    def test_cli_rejects_ambiguous_base_and_protected_output(self):
        for argv in (["--cluster-config", "config/cluster-4k.json", "tuning", "show", "tp4"],
                     ["tuning", "create", "tp4", "--nodes-config", "config/examples/nodes-ring4.json",
                      "--output", "config/new.json"]):
            args = spark3.parser().parse_args(argv)
            with self.assertRaises(SystemExit):
                args.func(args)


if __name__ == "__main__":
    unittest.main()
