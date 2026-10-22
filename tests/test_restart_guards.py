"""Host-independent tests for the guards on starting, stopping and syncing the cluster."""

from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
import subprocess
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)

TP4 = ["--tensor-parallel-size", "4", "--max-model-len", "1048576", "--kv-cache-memory-bytes", "11274289152",
       "--nnodes", "4"]
TP3 = ["--tensor-parallel-size", "3", "--max-model-len", "524288", "--kv-cache-memory-bytes", "3758096384",
       "--nnodes", "3"]
NODES3 = {"ssh_user": "u", "nodes": [
    {"name": "dgx1", "head": True, "rank": 0, "management_ip": "1", "roce_peer_hcas": {}},
    {"name": "dgx2", "head": False, "rank": 1, "management_ip": "2", "roce_peer_hcas": {}},
    {"name": "dgx3", "head": False, "rank": 2, "management_ip": "3", "roce_peer_hcas": {}}]}
CLUSTER = {"deployment": {"launch_enabled": True}, "container": {"name": "c", "image": "img-r6h"}}


def live(command: list[str], running: bool = True, profile: str | None = "config/cluster-tp4.json") -> dict:
    return {"shape": spark.engine_shape(command), "image": "img-r6h", "running": running, "profile": profile}


class NamedProfileTest(unittest.TestCase):
    def test_every_apply_needs_a_named_profile(self) -> None:
        for command in (spark.command_cluster_start, spark.command_cluster_stop,
                        spark.command_cluster_sync, spark.command_build_image):
            args = argparse.Namespace(apply=True, cluster_config=None, replace=True, remove=False,
                                      revision=None, tag=None)
            with self.subTest(command=command.__name__), self.assertRaises(SystemExit) as raised:
                command(args)
            self.assertIn("--cluster-config", str(raised.exception))

    def test_plans_and_read_only_commands_keep_the_default(self) -> None:
        spark.require_named_profile(argparse.Namespace(apply=False, cluster_config=None))
        self.assertEqual(spark.cluster_config_path(argparse.Namespace(cluster_config=None))[1],
                         spark.DEFAULT_CLUSTER_CONFIG)

    def test_the_parser_no_longer_fills_in_the_default(self) -> None:
        args = spark.parser().parse_args(["cluster", "start", "--apply"])
        self.assertIsNone(args.cluster_config)


class EngineShapeTest(unittest.TestCase):
    def test_shape_and_text(self) -> None:
        self.assertEqual(spark.engine_text(spark.engine_shape(TP4)),
                         "4 nodes, TP4, context 1,048,576, KV 10.5 GiB")

    def test_single_node_commands_default_to_one(self) -> None:
        shape = spark.engine_shape(["--max-model-len", "4096"])
        self.assertEqual((shape["nodes"], shape["tp"], shape["kv"]), ("1", "1", None))


class StartRefusalTest(unittest.TestCase):
    def test_a_profile_with_fewer_nodes_than_the_running_engine_is_refused(self) -> None:
        refusal = spark.start_refusal(live(TP4), spark.engine_shape(TP3), change_shape=True)
        self.assertIn("4-node engine is running", refusal)
        self.assertIn("bin/spark --cluster-config config/cluster-tp4.json cluster stop --apply", refusal)

    def test_a_same_shape_restart_needs_nothing_more(self) -> None:
        self.assertIsNone(spark.start_refusal(live(TP4), spark.engine_shape(TP4), change_shape=False))

    def test_a_shape_change_over_a_stopped_engine_needs_its_flag(self) -> None:
        refusal = spark.start_refusal(live(TP4, running=False), spark.engine_shape(TP3), change_shape=False)
        self.assertIn("--change-shape", refusal)
        self.assertIn("nodes, tp, context, kv", refusal)
        self.assertIsNone(spark.start_refusal(live(TP4, running=False), spark.engine_shape(TP3), change_shape=True))

    def test_a_kv_change_on_the_same_nodes_needs_its_flag(self) -> None:
        smaller = [*TP4[:5], "8000000000", *TP4[6:]]
        refusal = spark.start_refusal(live(TP4), spark.engine_shape(smaller), change_shape=False)
        self.assertIn("changes the engine's kv", refusal)

    def test_no_engine_means_a_new_start(self) -> None:
        self.assertIsNone(spark.start_refusal(None, spark.engine_shape(TP3), change_shape=False))


class StartAndStopTest(unittest.TestCase):
    def run_command(self, command, live_engine, *, apply: bool, **extra) -> tuple[int, list, str]:
        calls = []

        def run_ssh(_nodes, node, *argv):
            calls.append((node["name"], argv[:2]))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        args = argparse.Namespace(apply=apply, cluster_config="config/cluster.json", replace=True, remove=False,
                                  parallel=False, change_shape=False, verbose=False, **extra)
        out = io.StringIO()
        with mock.patch.object(spark, "configuration", return_value=(CLUSTER, NODES3, {})), \
                mock.patch.object(spark, "expected_command", return_value=TP3), \
                mock.patch.object(spark, "live_engine", return_value=live_engine), \
                mock.patch.object(spark, "run_ssh", side_effect=run_ssh), \
                mock.patch.object(spark, "require_local_deployment_state",
                                  side_effect=AssertionError("preflight reached")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = command(args)
        return code, calls, out.getvalue()

    def test_start_refuses_a_three_node_profile_over_a_running_four_node_engine(self) -> None:
        code, calls, out = self.run_command(spark.command_cluster_start, live(TP4), apply=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("4-node engine is running", out)

    def test_the_plan_shows_the_change_and_the_refusal(self) -> None:
        code, _, out = self.run_command(spark.command_cluster_start, live(TP4), apply=False)
        self.assertEqual(code, 0)
        self.assertIn("live:    running: 4 nodes, TP4, context 1,048,576, KV 10.5 GiB", out)
        self.assertIn("plan:    3 nodes, TP3, context 524,288, KV 3.5 GiB", out)
        self.assertIn("shape:   CHANGES nodes, tp, context, kv", out)
        self.assertIn("--apply would refuse", out)
        self.assertNotIn("docker run", out)

    def test_stop_refuses_a_profile_that_covers_part_of_the_running_engine(self) -> None:
        code, calls, out = self.run_command(spark.command_cluster_stop, live(TP4), apply=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertIn("spans 4 nodes and this profile covers 3", out)

    def test_stop_of_the_engines_own_profile_goes_ahead(self) -> None:
        with mock.patch.object(spark, "stop_memguard"), mock.patch.object(spark, "memguard_is_active",
                                                                          return_value=False):
            code, calls, _ = self.run_command(spark.command_cluster_stop, live(TP3), apply=True)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(node for node, argv in calls if argv == ("docker", "stop")), ["dgx1", "dgx2", "dgx3"])


class LaunchLabelTest(unittest.TestCase):
    def test_containers_record_the_profile_they_started_from(self) -> None:
        with mock.patch.object(spark, "rendered_docker_command",
                               return_value=["docker", "run", "-d", "--name", "c", "--network=host", "img"]):
            command = spark.launch_command(CLUSTER, NODES3["nodes"][0], "config/cluster-tp4.json")
        self.assertEqual(command[:7], ["docker", "run", "-d", "--name", "c", "--label",
                                       "spark.cluster-config=config/cluster-tp4.json"])


class CableSubnetTest(unittest.TestCase):
    HCAS = ["rocep1s0f0", "rocep1s0f1"]

    def test_a_multi_node_start_needs_every_interfaces_subnet(self) -> None:
        node = {"name": "dgx1", "roce_subnets": {"rocep1s0f0": "10.12.1.0/24"}}
        problems = spark.unrecorded_subnet_problems(NODES3, node, self.HCAS)
        self.assertEqual(len(problems), 1)
        self.assertIn("rocep1s0f1", problems[0])

    def test_complete_maps_and_single_nodes_pass(self) -> None:
        node = {"name": "dgx1", "roce_subnets": {"rocep1s0f0": "10.12.1.0/24", "rocep1s0f1": "10.13.1.0/24"}}
        self.assertEqual(spark.unrecorded_subnet_problems(NODES3, node, self.HCAS), [])
        single = {"nodes": [NODES3["nodes"][0]]}
        self.assertEqual(spark.unrecorded_subnet_problems(single, {"name": "dgx1"}, self.HCAS), [])

    def test_the_example_site_map_records_its_subnets(self) -> None:
        example = spark.json.loads((ROOT / "config" / "nodes.example.json").read_text())
        for node in example["nodes"]:
            hcas = sorted({hca for peers in node["roce_peer_hcas"].values() for hca in peers})
            self.assertEqual(spark.unrecorded_subnet_problems(example, node, hcas), [])


if __name__ == "__main__":
    unittest.main()


class SyncRevisionTest(unittest.TestCase):
    """A named revision is resolved after a fetch, so a commit merged since the last fetch
    (or origin/main) deploys instead of failing or detaching at a stale commit."""

    def test_fetch_before_resolving(self):
        calls = []
        args = argparse.Namespace(revision="origin/main", apply=False)
        with mock.patch.object(spark, "require_named_profile"), \
                mock.patch.object(spark, "configuration", return_value=({"x": 1}, {"nodes": [{"name": "n1"}]}, {})), \
                mock.patch.object(spark, "cluster_config_path", return_value=(None, "config/c.json")), \
                mock.patch.object(spark, "repository_path", return_value="/p"), \
                mock.patch.object(spark, "checked", side_effect=lambda command, **kw: calls.append(command[1])), \
                mock.patch.object(spark, "repository_revision", side_effect=lambda r: calls.append(f"resolve {r}") or "abc"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(spark.command_cluster_sync(args), 0)
        self.assertEqual(calls, ["fetch", "resolve origin/main"])



class UnqualifiedShapeTest(unittest.TestCase):
    """A recipe shape starts in production only once a release benchmark qualifies its image."""

    def start(self, cluster: dict, unqualified_shape: bool = False) -> str:
        args = argparse.Namespace(apply=True, cluster_config="config/cluster.json", shape=None, replace=True,
                                  remove=False, parallel=False, change_shape=False, verbose=False,
                                  unqualified_shape=unqualified_shape)

        def no_ssh(*argv, **kwargs):
            raise AssertionError("contacted a node")

        with mock.patch.object(spark, "configuration", return_value=(cluster, NODES3, {})), \
                mock.patch.object(spark, "expected_command", return_value=TP3), \
                mock.patch.object(spark, "live_engine", return_value=None), \
                mock.patch.object(spark, "run_ssh", side_effect=no_ssh), \
                mock.patch.object(spark, "require_local_deployment_state",
                                  side_effect=AssertionError("preflight reached")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                spark.command_cluster_start(args)
            except (SystemExit, AssertionError) as stop:
                return str(stop)
        return "started"

    def test_an_unqualified_shape_is_refused_unless_overridden(self) -> None:
        # The 4 KiB-page triangle and the ring were not benched with spark-ds41f-1022's image.
        for shape in ("tp3-triangle-4k", "tp4-ring"):
            unqualified = spark.cluster_recipe.resolve(spark.read_json("config/cluster.json"), shape)
            self.assertNotIn("benchmark_reference", unqualified)
            self.assertIn(f"no release benchmark qualifies this recipe's image on {shape}", self.start(unqualified))
            self.assertEqual(self.start(unqualified, unqualified_shape=True), "preflight reached")
        qualified = spark.cluster_recipe.resolve(spark.read_json("config/cluster.json"), "tp3-triangle")
        self.assertEqual(self.start(qualified), "preflight reached")
        # The frozen r6n recipe still starts the ring it was released on.
        frozen = spark.cluster_recipe.resolve(spark.read_json("experiments/2026-10-16-r6s/r6n/cluster.json"), "tp4-ring")
        self.assertEqual(self.start(frozen), "preflight reached")

    def test_a_concrete_profile_has_no_qualification_check(self) -> None:
        archived = spark.read_json("experiments/2026-10-14-one-recipe/tp3-r6h-production.json")
        self.assertNotIn("shape", archived)
        self.assertEqual(self.start(archived), "preflight reached")
