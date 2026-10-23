"""Names before and after the rename to spark-ds41f (the release after spark-ds41f-1022).

Images built before it carry ``local.spark3.*`` labels and serve as ``dsv41-karmic-kraken``;
code reads either label prefix, guards recognise either container, and a start never runs
beside an engine serving under the other name."""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_names_test", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)
source_labels = spark.source_labels

TREE = "e68aa19c8027c34a3dcfeb3ed203f6d285fc4d67"
FROZEN_1022 = "experiments/2026-10-23-next-release/1022/cluster.json"
NODES = {"ssh_user": "u", "nodes": [{"name": "dgx1", "head": True, "rank": 0, "management_ip": "1"}]}


def relabeled(labels: dict) -> dict:
    """``labels`` as an image built after the rename carries them."""
    return {key.replace(source_labels.LEGACY_PREFIX, source_labels.PREFIX, 1): value
            for key, value in labels.items()}


class SourceLabelTest(unittest.TestCase):
    def test_either_prefix_reads(self) -> None:
        self.assertEqual(source_labels.get({"local.spark3.vllm.tree": TREE}, "vllm.tree"), TREE)
        self.assertEqual(source_labels.get({"local.spark-ds41f.vllm.tree": TREE}, "vllm.tree"), TREE)
        self.assertIsNone(source_labels.get({}, "vllm.tree"))
        self.assertIsNone(source_labels.get({"local.spark3.b12x.tree": TREE}, "vllm.tree"))

    def test_the_current_name_wins_and_names_what_is_missing(self) -> None:
        both = {"local.spark3.vllm.tree": "old", "local.spark-ds41f.vllm.tree": "new"}
        self.assertEqual(source_labels.get(both, "vllm.tree"), "new")
        self.assertEqual(source_labels.key({"local.spark3.vllm.tree": TREE}, "vllm.tree"), "local.spark3.vllm.tree")
        self.assertEqual(source_labels.key({}, "vllm.tree"), "local.spark-ds41f.vllm.tree")

    def test_a_profile_with_the_new_labels_passes_the_same_source_checks(self) -> None:
        """spark-ds41f-1022's checks of its image trees give the same answer under either prefix."""
        args = type("Args", (), {"cluster_config": FROZEN_1022, "shape": "tp4-ring"})()
        cluster, nodes, upstreams = spark.configuration(args)
        renamed = copy.deepcopy(cluster)
        renamed["container"]["expected_labels"] = relabeled(cluster["container"]["expected_labels"])
        self.assertEqual(spark.local_doctor(renamed, nodes, upstreams), spark.local_doctor(cluster, nodes, upstreams))
        manifest = spark.read_json(upstreams["source_manifest"])
        self.assertEqual(spark.kernel_backend.source_problems(renamed, upstreams, manifest), [])
        wrong = copy.deepcopy(renamed)
        wrong["container"]["expected_labels"]["local.spark-ds41f.vllm.tree"] = "0" * 40
        self.assertTrue(any("local.spark-ds41f.vllm.tree" in problem
                            for problem in spark.kernel_backend.source_problems(wrong, upstreams, manifest)))


class ServingContainerTest(unittest.TestCase):
    OLD = {"container": {"name": "dsv41-karmic-kraken"}}
    NEW = {"container": {"name": "spark-ds41f"}}

    def test_every_serving_name_is_recognised_whatever_the_profile_calls_its_own(self) -> None:
        for cluster in (self.OLD, self.NEW):
            for name in ("spark-ds41f", "dsv41-karmic-kraken", "dsv41-other"):
                self.assertTrue(spark.serving_container(name, cluster), (name, cluster))
            for name in ("spark-ds41f-pre-1760000000-x", "tilelang-build", "dsv4-other"):
                self.assertFalse(spark.serving_container(name, cluster), (name, cluster))
        self.assertTrue(spark.serving_container("custom", {"container": {"name": "custom"}}))

    def test_docker_filters_name_every_serving_container(self) -> None:
        self.assertEqual(spark.serving_filters(self.NEW),
                         ["--filter", "name=^spark-ds41f$", "--filter", "name=^dsv41-"])
        self.assertEqual(spark.serving_filters(self.OLD),
                         ["--filter", "name=^dsv41-karmic-kraken$", "--filter", "name=^spark-ds41f$",
                          "--filter", "name=^dsv41-"])

    def test_a_build_refuses_either_service(self) -> None:
        for running in ("spark-ds41f\n", "dsv41-karmic-kraken\n"):
            for cluster in (self.OLD, self.NEW):
                ps = subprocess.CompletedProcess([], 0, stdout=f"other\n{running}")
                with self.subTest(running=running, profile=cluster["container"]["name"]), \
                        mock.patch.object(spark.subprocess, "run", return_value=ps), \
                        self.assertRaises(SystemExit):
                    spark.require_idle_build_host(cluster)
        with mock.patch.object(spark.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="x\n")):
            spark.require_idle_build_host(self.NEW)


def start_problems(cluster: dict, ps: str, image_labels: dict | None = None) -> list[str]:
    """A start's preflight on one node: ``ps`` is what ``docker ps`` lists (name, tab, profile)
    and the image carries ``image_labels`` (default: the profile's expected labels)."""
    labels = json.dumps([{"Config": {"Labels": cluster["container"]["expected_labels"]
                                     if image_labels is None else image_labels}}])

    def run_ssh(nodes, node, *command):
        if command[:2] == ("docker", "ps"):
            return subprocess.CompletedProcess(command, 0, ps, "")
        if command[:3] == ("docker", "image", "inspect"):
            return subprocess.CompletedProcess(command, 0, labels, "")
        if command[:2] == ("docker", "inspect"):
            return subprocess.CompletedProcess(command, 1, "", "no such container")
        return subprocess.CompletedProcess(command, 0, "", "")

    cluster = copy.deepcopy(cluster)
    cluster["host"].pop("kernel", None)
    with mock.patch.object(spark, "run_ssh", side_effect=run_ssh), \
            mock.patch.object(spark, "remote_repository_state",
                              return_value=("abc", cluster["deployment"]["repository"], False)), \
            mock.patch.object(spark, "roce_gid_setting", return_value=(3, [])), \
            mock.patch.object(spark, "remote_shell", return_value=subprocess.CompletedProcess([], 0, "", "")), \
            mock.patch.object(spark, "roce_gid_problems", return_value=[]), \
            mock.patch.object(spark.topology, "gid_subnet_problems", return_value=[]), \
            mock.patch.object(spark, "unrecorded_subnet_problems", return_value=[]):
        return spark.node_runtime_problems(cluster, NODES, NODES["nodes"][0], "abc",
                                           allow_existing=True, cluster_config="arm.json")


class StartBesideAnotherEngineTest(unittest.TestCase):
    """``cluster start --replace`` replaces only its own container name: an engine serving
    under the other name must be stopped with its own profile first."""

    def setUp(self) -> None:
        self.old = spark.read_cluster("experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp4.json")
        self.new = copy.deepcopy(self.old)
        self.new["container"]["name"] = "spark-ds41f"

    def test_a_start_beside_the_other_name_is_refused_with_its_stop_command(self) -> None:
        problems = start_problems(self.new, "dsv41-karmic-kraken\texperiments/2026-10-23-next-release/1022/cluster.json\n")
        self.assertEqual(problems, [
            "dgx1: dsv41-karmic-kraken is serving; stop it with its own profile before starting spark-ds41f: "
            "bin/spark --cluster-config experiments/2026-10-23-next-release/1022/cluster.json cluster stop --apply"])
        problems = start_problems(self.old, "spark-ds41f\t\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("spark-ds41f is serving", problems[0])
        self.assertIn("--cluster-config PATH", problems[0])

    def test_the_profiles_own_container_is_left_to_replace(self) -> None:
        self.assertEqual(start_problems(self.new, "spark-ds41f\tconfig/cluster.json\n"), [])
        self.assertEqual(start_problems(self.old, "dsv41-karmic-kraken\tconfig/cluster.json\n"), [])
        self.assertEqual(start_problems(self.new, ""), [])


if __name__ == "__main__":
    unittest.main()
