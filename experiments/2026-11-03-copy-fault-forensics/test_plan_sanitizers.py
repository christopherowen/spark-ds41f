"""CPU-only checks for a planner that must never launch its supplied commands."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("plan_sanitizers.py")


class SanitizerPlanTest(unittest.TestCase):
    def plan(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             "--sanitizer-path", "/not-installed/compute-sanitizer",
             "--source-root", "/not-present/candidate37",
             "--report-dir", "/not-created/reports", *extra],
            text=True, capture_output=True, check=True,
        )

    def test_prints_ordered_plan_without_requiring_gpu_or_installed_tools(self):
        plan = json.loads(self.plan().stdout)
        self.assertTrue(plan["plan_only"])
        self.assertFalse(plan["commands_executed"])
        jobs = plan["jobs"]
        self.assertEqual([job["tool"] for job in jobs], ["memcheck"] * 4 + ["initcheck"] * 4)
        for job in jobs:
            argv = job["argv"]
            self.assertEqual(argv[argv.index("--error-exitcode") + 1], "86")
            self.assertEqual(argv[argv.index("--target-processes") + 1], "all")
            self.assertNotIn("--force-blocking-launches", argv)
            self.assertEqual(job["env_unset"], ["CUDA_LAUNCH_BLOCKING"])
            self.assertNotIn("CUDA_LAUNCH_BLOCKING", job["env_set"])
            if job["tool"] == "initcheck":
                self.assertTrue(job["requires"])
        self.assertIn("Declared", plan["toolchain"]["version_status"])

    def test_serving_argv_is_preserved_but_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')",
                       "literal space", "$(not-a-command)", "; also literal"]
            plan = json.loads(self.plan("--serving-command", *command).stdout)
            self.assertFalse(marker.exists())
        serving = [job for job in plan["jobs"] if job["id"].endswith("serving-repro")]
        self.assertEqual(len(serving), 2)
        for job in serving:
            self.assertEqual(job["argv"][-len(command):], command)
            self.assertIn("Remote workers", job["limitation"])

    def test_empty_serving_command_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.plan("--serving-command")
        self.assertIn("needs an executable", caught.exception.stderr)

    def test_exported_image_identity_does_not_require_git_metadata(self):
        image_id = "sha256:" + "1" * 64
        source = json.loads(self.plan("--image-id", image_id).stdout)["source"]
        exported = source["identity_routes"]["exported_image"]
        self.assertEqual(exported["image_id"], image_id)
        self.assertEqual(exported["inspect_argv"][-1], image_id)
        self.assertIn("Git commands are unnecessary", exported["applicable_when"])
        hashes = source["focused_source_hashes"]
        self.assertEqual(hashes["argv"][0], "sha256sum")
        self.assertIn("vllm/models/deepseek_v4_1/tilelang/router.py", hashes["expected_sha256"])
        self.assertEqual(exported["expected_labels"]["local.spark-ds41f.vllm.tree"],
                         source["expected_vllm_tree"])

    def test_mutable_image_tag_is_not_identity_evidence(self):
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.plan("--image-id", "spark-ds41f:candidate-37")
        self.assertIn("immutable sha256 image ID", caught.exception.stderr)


if __name__ == "__main__":
    unittest.main()
