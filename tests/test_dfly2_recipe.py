"""Unit tests for standalone speculative drafter support (drafters/drafter) and DFly2 arms."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kernel_backend  # noqa: E402
import recipe  # noqa: E402
import release_tests  # noqa: E402
import spark_ds41f as spark  # noqa: E402
from spark_ds41f import config as spark_config  # noqa: E402
from spark_ds41f import doctor as spark_doctor  # noqa: E402


class DFly2RecipeAndDoctorTest(unittest.TestCase):
    def test_dfly2_tp4_arm_reuses_shared_hub_mount_and_preserves_tilelang_backend(self):
        base = recipe.resolve(spark_config.read_json("releases/spark-ds41f-1126/cluster.json"), "tp4-ring")
        arm = spark_config.read_cluster("experiments/2026-10-26-spark-ds41f-candidate/candidate-75/tp4.json")

        # Shared /hub root must not create a duplicate /hub or /drafter-hub bind mount.
        mount_targets = [m[1] for m in arm["container"]["mounts"]]
        self.assertEqual(mount_targets.count("/hub"), 1)
        self.assertFalse(any(t in recipe.DRAFTER_MOUNTS for t in mount_targets))

        args = arm["serve_args"]
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["method"], "dfly2")
        self.assertEqual(
            spec["model"],
            "/hub/models--AngelSlim--DS-V4.1-Flash-DFly2-B6/snapshots/d08e55a6897b1974307ea9771038838de5986e66",
        )
        self.assertEqual(spec["num_speculative_tokens"], 5)
        self.assertEqual(spec["draft_tensor_parallel_size"], 4)
        self.assertEqual(spec["attention_backend"], "TILELANG")
        self.assertTrue(spec["enable_adaptive_verification"])
        self.assertEqual(spec["adaptive_verification_row_cost"], 0.035)
        self.assertEqual(kernel_backend.problems(arm), [])
        self.assertEqual(spark_config.drafter_problems(arm), [])

        # Apart from --speculative-config, serve_args match 1126 tp4-ring byte-for-byte.
        base_args = list(base["serve_args"])
        arm_args = list(arm["serve_args"])
        base_args[base_args.index("--speculative-config") + 1] = "<spec>"
        arm_args[arm_args.index("--speculative-config") + 1] = "<spec>"
        self.assertEqual(arm_args, base_args)

    def test_all_six_dfly2_shape_arms_pass_local_doctor_and_preserve_shape_curves(self):
        arms = {
            "tp4": ("tp4-ring", 4, 1, 0.035),
            "tp3": ("tp3-triangle", 3, 1, 0.049),
            "tp3-triangle-4k": ("tp3-triangle-4k", 3, 1, 0.049),
            "tp2-direct": ("tp2-direct", 2, 1, 0.25),
            "tp1": ("tp1", 1, 1, 0.5),
            "pp2-tp2-ring": ("pp2-tp2-ring", 2, 2, 0.25),
        }
        upstreams = spark_config.read_json("experiments/2026-10-26-spark-ds41f-candidate/candidate-75/upstreams.lock.json")
        for stem, (_shape, expected_tp, expected_pp, expected_row_cost) in arms.items():
            with self.subTest(stem=stem):
                cluster = spark_config.read_cluster(f"experiments/2026-10-26-spark-ds41f-candidate/candidate-75/{stem}.json")
                args = cluster["serve_args"]
                spec = json.loads(args[args.index("--speculative-config") + 1])
                self.assertEqual(spec["method"], "dfly2")
                self.assertEqual(spec["draft_tensor_parallel_size"], expected_tp)
                self.assertEqual(spec["adaptive_verification_row_cost"], expected_row_cost)
                if expected_pp > 1:
                    self.assertEqual(args[args.index("--pipeline-parallel-size") + 1], str(expected_pp))
                example_rel = spark_config.NODE_MAP_EXAMPLES[cluster["nodes_config"]]
                nodes = spark_config.read_json(example_rel)
                head = next(n for n in nodes["nodes"] if n.get("head"))
                cluster["distributed"]["master_addr"] = head.get("control_ip") or head["management_ip"]
                findings = spark_doctor.local_doctor(cluster, nodes, upstreams)
                errors, warnings = spark_doctor.split_findings(findings)
                self.assertEqual(errors, [], f"{stem} errors: {errors}")
                self.assertEqual(warnings, [], f"{stem} warnings: {warnings}")

    def test_drafter_separate_hub_or_snapshot_mounts_and_unset_overlay(self):
        base = spark_config.read_cluster("experiments/2026-10-26-spark-ds41f-candidate/candidate-75/tp4.json")
        # Separate hub root mounts /drafter-hub without colliding with /hub
        custom_hub = recipe.overlay(
            {
                "extends": {},
                "set": {
                    "drafters": {
                        "dfly2-alt": {
                            "method": "dfly2",
                            "hub": "/mnt/nvme/hf-hub",
                            "snapshot": "models--AngelSlim--DS-V4.1-Flash-DFly2-B6/snapshots/d08e55a6897b1974307ea9771038838de5986e66",
                            "drafts": 7,
                        }
                    },
                    "drafter": "dfly2-alt",
                },
            },
            base,
        )
        mounts = {m[1]: m for m in custom_hub["container"]["mounts"]}
        self.assertIn("/drafter-hub", mounts)
        args = custom_hub["serve_args"]
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["num_speculative_tokens"], 7)
        # 32 streams * (1 + 7 drafts) = 256 rows
        self.assertEqual(args[args.index("--max-cudagraph-capture-size") + 1], "256")
        self.assertEqual(recipe.capture_problems(custom_hub), ([], []))
        self.assertEqual(spark_config.drafter_problems(custom_hub), [])

        # Unsetting drafter restores built-in DSpark speculative config cleanly
        reverted = recipe.overlay({"extends": {}, "unset": ["drafter"]}, custom_hub)
        rev_mounts = {m[1] for m in reverted["container"]["mounts"]}
        self.assertNotIn("/drafter-hub", rev_mounts)
        rev_spec = json.loads(
            reverted["serve_args"][reverted["serve_args"].index("--speculative-config") + 1]
        )
        self.assertEqual(rev_spec["method"], "dspark")
        self.assertNotIn("model", rev_spec)

    def test_node_drafter_doctor_and_weight_fragmentation_include_drafter(self):
        cluster = copy.deepcopy(spark_config.read_cluster("experiments/2026-10-26-spark-ds41f-candidate/candidate-75/tp4.json"))
        cluster["host"]["home"] = "/home/swank"
        nodes = {"ssh_user": "swank", "nodes": [{"name": "dgx1", "rank": 0, "head": True}]}
        ok_payload = json.dumps(
            {
                "config_sha256": "0d53324203ab18c69927da96aa27566195602b9ab0aac4867b2819d6a46a051e",
                "index_sha256": "7192e8294c56f3a9064d7b3dfa4494dc7486009f3229e98e9c6d0d2fbdaf6786",
                "missing_shards": [],
            }
        )
        with mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(["ssh"], 0, stdout=ok_payload, stderr=""),
        ):
            self.assertEqual(spark_config.node_drafter_problems(cluster, nodes, nodes["nodes"][0]), [])

        bad_payload = json.dumps(
            {
                "config_sha256": "0" * 64,
                "index_sha256": "7192e8294c56f3a9064d7b3dfa4494dc7486009f3229e98e9c6d0d2fbdaf6786",
                "missing_shards": ["model-00004-of-00004.safetensors"],
            }
        )
        with mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(["ssh"], 0, stdout=bad_payload, stderr=""),
        ):
            errs = spark_config.node_drafter_problems(cluster, nodes, nodes["nodes"][0])
            self.assertEqual(len(errs), 2)
            self.assertIn("config.json has sha256", errs[0])
            self.assertIn("model-00004-of-00004.safetensors", errs[1])

        captured_cmds = []

        def fake_run(cmd, **_kwargs):
            captured_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"shards": []}), stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            findings = spark_doctor.weight_fragmentation_findings(cluster, nodes)
        self.assertEqual(findings, [])
        self.assertEqual(len(captured_cmds), 1)
        remote_cmd = captured_cmds[0][-1]
        self.assertIn("models--deepseek-ai--DeepSeek-V4.1-Flash-1017", remote_cmd)
        self.assertIn("models--AngelSlim--DS-V4.1-Flash-DFly2-B6", remote_cmd)

    def test_release_tests_stack_matches_dfly2_and_config_is_valid(self):
        self.assertTrue(release_tests.STACK.search("tests/models/deepseek_v4_1/test_dfly2.py"))
        self.assertTrue(release_tests.STACK.search("tests/v1/spec_decode/test_dfly2_speculator.py"))
        cfg = release_tests.load_config(ROOT / "config/release-tests.json")
        self.assertEqual(
            release_tests.classify("tests/models/deepseek_v4_1/test_dfly2.py", cfg),
            ["gate"],
        )


if __name__ == "__main__":
    unittest.main()
