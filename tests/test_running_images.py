"""Validate the images actually serving, including Docker's two ID representations."""

import contextlib
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark


class RunningImageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config_path = Path(temporary.name) / "arm.json"
        self.config_path.write_text("{}")
        self.labels = {"local.spark-ds41f.vllm.tree": "a" * 40}
        self.cluster = {
            "promoted_baseline": "2026-10-30-spark-ds41f-1030",
            "container": {"name": "service", "image": "spark-ds41f:candidate",
                          "expected_labels": self.labels},
        }
        self.nodes = {"nodes": [{"name": "dgx1"}, {"name": "dgx2"}]}
        self.ids = {"dgx1": "sha256:" + "1" * 64, "dgx2": "sha256:" + "2" * 64}
        self.images = {name: {
            "Id": image_id,
            "RootFS": {"Layers": ["sha256:" + "3" * 64, "sha256:" + "4" * 64]},
            "Config": {"Env": ["PATH=/image/bin"], "Labels": dict(self.labels)},
        } for name, image_id in self.ids.items()}
        self.failed_images = set()
        self.failed_containers = set()
        self.calls = []

    def run_ssh(self, nodes, node, *command):
        name = node["name"]
        self.calls.append((name, command))
        if command[:3] == ("docker", "image", "inspect"):
            if command[3] == self.cluster["container"]["image"]:
                # A moved tag can point to the correct build while the old one serves.
                image = {**self.images[name], "Config": {"Labels": self.labels}}
            else:
                self.assertEqual(command[3], self.ids[name])
                if name in self.failed_images:
                    return subprocess.CompletedProcess(command, 1, "", "image unavailable")
                image = self.images[name]
            output = json.dumps([image])
        elif command[:2] == ("docker", "inspect"):
            if name in self.failed_containers:
                return subprocess.CompletedProcess(command, 1, "", "no such container")
            if command[2] == "--format":
                output = f"{self.ids[name]} 2026-11-03T00:00:00Z"
            else:
                output = json.dumps([{
                    "Image": self.ids[name], "State": {"Running": True},
                    "Config": {"Image": self.cluster["container"]["image"],
                               "Cmd": ["serve"], "Env": ["PATH=/image/bin"]},
                    "HostConfig": {"Binds": []},
                }])
        else:
            self.assertIn(command[0], ("efibootmgr", "git"))
            output = "a" * 40
        return subprocess.CompletedProcess(command, 0, output, "")

    def doctor(self):
        with contextlib.ExitStack() as stack:
            for name in (
                "site_node_problems", "desktop_problems", "boot_order_problems", "modeset_problems",
                "fbdev_problems", "drm_card_problems", "drm_master_problems", "console_mode_problems",
                "splash_problems", "idle_service_warnings", "nvme_coalescing_warnings",
                "nvme_apst_warnings", "wireless_warnings", "fan_control_problems",
                "roce_gid_findings", "node_glp_problems", "clock_latch_problems",
            ):
                stack.enter_context(mock.patch.object(spark, name, return_value=[]))
            stack.enter_context(mock.patch.object(spark, "kernel_policy_path",
                                                  return_value=self.config_path.with_name("absent.json")))
            stack.enter_context(mock.patch.object(spark, "carveout_weights_enabled", return_value=False))
            stack.enter_context(mock.patch.object(spark, "roce_gid_setting", return_value=(None, [])))
            stack.enter_context(mock.patch.object(spark, "remote_shell",
                                                  return_value=subprocess.CompletedProcess([], 0, "", "")))
            stack.enter_context(mock.patch.object(spark, "run_ssh", side_effect=self.run_ssh))
            stack.enter_context(mock.patch.object(spark, "expected_command", return_value=["serve"]))
            stack.enter_context(mock.patch.object(spark, "expected_environment", return_value={}))
            stack.enter_context(mock.patch.object(spark, "expected_binds", return_value=[]))
            stack.enter_context(mock.patch.object(spark, "configured_cpuset", return_value=None))
            return spark.live_doctor(self.cluster, self.nodes)

    def bench(self):
        with (
            mock.patch.object(spark, "run_ssh", side_effect=self.run_ssh),
            mock.patch.object(spark, "live_doctor", return_value=[]),
            mock.patch.object(spark, "repository_revision", return_value="a" * 40),
            mock.patch.object(spark, "repository_dirty", return_value=""),
            mock.patch.object(spark, "repository_path", return_value="/deployment"),
            mock.patch.object(spark.topology, "transport", return_value="oneshot-direct"),
            mock.patch.object(spark.kernel_backend, "backend", return_value="tilelang"),
        ):
            return spark.bench_identity(self.cluster, self.nodes, "arm.json", self.config_path)

    def test_faithful_copies_with_different_store_ids_pass(self):
        self.assertEqual(self.doctor(), [])
        identity = self.bench()
        self.assertEqual(identity["live_problems"], [])
        first, second = (identity["nodes"][name] for name in self.ids)
        self.assertNotEqual(first["image_id"], second["image_id"])
        self.assertEqual(first["image_content_sha256"], second["image_content_sha256"])
        self.assertEqual(first["image_layers"], self.images["dgx1"]["RootFS"]["Layers"])
        self.assertEqual(first["image_labels"], self.labels)

    def test_moved_tag_cannot_hide_wrong_running_source(self):
        self.images["dgx2"]["Config"]["Labels"] = {"local.spark-ds41f.vllm.tree": "b" * 40}
        for problems in (self.doctor(), self.bench()["live_problems"]):
            self.assertTrue(any("dgx2: running image label" in p for p in problems), problems)
        self.assertFalse(any(command == ("docker", "image", "inspect", self.cluster["container"]["image"])
                             for _, command in self.calls))

    def test_same_labels_do_not_hide_different_layer_contents_or_order(self):
        original = copy.deepcopy(self.images["dgx2"]["RootFS"]["Layers"])
        for layers in (["sha256:" + "5" * 64], list(reversed(original))):
            with self.subTest(layers=layers):
                self.images["dgx2"]["RootFS"]["Layers"] = layers
                for problems in (self.doctor(), self.bench()["live_problems"]):
                    self.assertTrue(any("layers or runtime Config differ" in p for p in problems), problems)

    def test_same_layers_do_not_hide_changed_image_defaults(self):
        self.images["dgx2"]["Config"]["Env"].append("ALTER_SAMPLING=1")
        for problems in (self.doctor(), self.bench()["live_problems"]):
            self.assertTrue(any("layers or runtime Config differ" in p for p in problems), problems)

    def test_failed_image_inspection_fails_even_when_later_doctor_is_clean(self):
        self.failed_images.add("dgx2")
        for problems in (self.doctor(), self.bench()["live_problems"]):
            self.assertTrue(any("dgx2: cannot inspect running image" in p for p in problems), problems)

    def test_benchmark_cannot_record_missing_container_as_valid(self):
        self.failed_containers.add("dgx2")
        identity = self.bench()
        self.assertIsNone(identity["nodes"]["dgx2"]["image_id"])
        self.assertTrue(any("dgx2: cannot read" in p for p in identity["live_problems"]))

    def test_missing_image_metadata_fails_closed(self):
        original = copy.deepcopy(self.images["dgx2"])
        for field, value in (("RootFS", {}), ("RootFS", {"Layers": []}), ("Config", None)):
            with self.subTest(field=field, value=value):
                self.images["dgx2"] = {**original, field: value}
                self.assertTrue(any("dgx2: running image has no" in p for p in self.doctor()))
                self.assertTrue(any("dgx2: running image has no" in p for p in self.bench()["live_problems"]))

    def test_invalid_image_id_never_becomes_a_tag_inspection(self):
        self.ids["dgx2"] = "spark-ds41f:candidate"
        for problems in (self.doctor(), self.bench()["live_problems"]):
            self.assertTrue(any("dgx2: cannot read" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
