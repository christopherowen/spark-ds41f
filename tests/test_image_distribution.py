"""Content identity and fail-closed image inspection, without Docker or SSH."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spark_ds41f as spark


class ImageDistributionTest(unittest.TestCase):
    def setUp(self):
        self.cluster = {"container": {"image": "candidate", "expected_labels": {"source.tree": "wanted"}}}
        self.nodes = {"ssh_user": "user", "nodes": [
            {"name": "head", "head": True, "management_ip": "head"},
            {"name": "worker", "head": False, "management_ip": "worker"},
        ]}
        self.image = {"Id": "sha256:" + "1" * 64,
                      "RootFS": {"Layers": ["sha256:" + "a" * 64]},
                      "Config": {"Labels": {"source.tree": "wanted"}, "Env": ["MODE=good"]}}
        self.held = {node["name"]: copy.deepcopy(self.image) for node in self.nodes["nodes"]}
        self.held["worker"]["Id"] = "sha256:" + "2" * 64
        self.failures = {}
        self.inspections = []

    def inspect(self, _nodes, node, *command):
        self.assertEqual(command[:3], ("docker", "image", "inspect"))
        self.assertEqual(len(command), 4)
        self.inspections.append((node["name"], command[3]))
        return self.failures.get(node["name"], subprocess.CompletedProcess(
            command, 0, json.dumps([self.held[node["name"]]]), ""))

    def local(self, image=None):
        return mock.patch.object(spark.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], 0, json.dumps([image or self.image]), ""))

    def copies(self):
        with self.local(), mock.patch.object(spark, "run_ssh", side_effect=self.inspect):
            return spark.image_copies(self.cluster, self.nodes, "config/candidate.json")

    def test_different_store_ids_with_identical_content_need_no_copy(self):
        identity, problem, differ = self.copies()
        self.assertIsNone(problem)
        self.assertEqual(identity, spark.image_content_identity(self.image))
        self.assertEqual(differ, [])

    def test_equal_layers_with_different_source_labels_or_environment_need_repair(self):
        for field, changed in (("Labels", {"source.tree": "stale"}), ("Env", ["MODE=bad"])):
            with self.subTest(field=field):
                self.held["worker"] = copy.deepcopy(self.image)
                self.held["worker"]["Config"][field] = changed
                self.assertEqual([n["name"] for n in self.copies()[2]], ["worker"])

    def test_explicit_missing_image_is_repairable(self):
        self.failures["worker"] = subprocess.CompletedProcess([], 1, "", "Error: No such image: candidate")
        self.assertEqual([n["name"] for n in self.copies()[2]], ["worker"])

    def test_failed_or_malformed_remote_inspection_prevents_any_distribution(self):
        bad = [subprocess.CompletedProcess([], 255, "", "SSH timed out"),
               subprocess.CompletedProcess([], 1, "", "permission denied: docker.sock")]
        bad += [subprocess.CompletedProcess([], 0, output, "") for output in (
            "not-json", "[]", "{}", "[{}, {}]", "[null]",
            json.dumps([{"RootFS": {"Layers": []}, "Config": {}}]),
            json.dumps([{**self.image, "Config": {"Labels": "invalid"}}]),
        )]
        args = argparse.Namespace(apply=True, cluster_config="config/candidate.json", tag=None)
        for failure in bad:
            with self.subTest(output=failure.stdout, error=failure.stderr):
                self.failures["worker"] = failure
                with self.local(), mock.patch.object(spark, "configuration", return_value=(self.cluster, self.nodes, {})), \
                        mock.patch.object(spark, "cluster_config_path", return_value=(None, args.cluster_config)), \
                        mock.patch.object(spark, "run_ssh", side_effect=self.inspect), \
                        mock.patch.object(spark, "load_image") as load, self.assertRaises(SystemExit):
                    spark.command_build_distribute(args)
                load.assert_not_called()

    def test_malformed_or_failed_local_inspection_is_not_a_missing_build_warning(self):
        for result in (subprocess.CompletedProcess([], 0, "[]", ""),
                       subprocess.CompletedProcess([], 0, "broken", ""),
                       subprocess.CompletedProcess([], 1, "", "Cannot connect to Docker daemon")):
            with self.subTest(result=result), mock.patch.object(spark.subprocess, "run", return_value=result), \
                    self.assertRaises(SystemExit):
                spark.local_image(self.cluster, "candidate", "config/candidate.json")

    def test_explicit_other_tag_keeps_label_exemption_but_requires_its_complete_content(self):
        other = copy.deepcopy(self.image)
        other["Config"]["Labels"] = {"source.tree": "another-source"}
        self.held = {node["name"]: copy.deepcopy(other) for node in self.nodes["nodes"]}
        self.held["worker"]["Config"]["Env"] = ["MODE=stale"]
        with self.local(other), mock.patch.object(spark, "run_ssh", side_effect=self.inspect):
            identity, problem, differ = spark.image_copies(self.cluster, self.nodes, "config/candidate.json", "other")
        self.assertIsNone(problem)
        self.assertEqual(identity, spark.image_content_identity(other))
        self.assertEqual([n["name"] for n in differ], ["worker"])
        self.assertEqual(set(self.inspections), {("head", "other"), ("worker", "other")})

    def test_a_successful_load_still_requires_the_full_expected_content(self):
        wanted = spark.image_content_identity(self.image)
        for changed in (False, True):
            with self.subTest(changed=changed):
                self.held["worker"]["Config"]["Env"] = ["MODE=bad"] if changed else ["MODE=good"]
                with mock.patch.object(spark.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "loaded", "")), \
                        mock.patch.object(spark, "run_ssh", side_effect=self.inspect):
                    failure = spark.load_image(self.nodes, self.nodes["nodes"][1], "candidate", wanted)
                if changed:
                    self.assertIn("image content", failure)
                else:
                    self.assertIsNone(failure)

    def test_failed_load_does_not_claim_verification(self):
        with mock.patch.object(spark.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "load failed")), \
                mock.patch.object(spark, "run_ssh") as inspect:
            failure = spark.load_image(self.nodes, self.nodes["nodes"][1], "candidate", spark.image_content_identity(self.image))
        self.assertIn("load failed", failure)
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
