"""Host-independent tests for the guards on starting, stopping and syncing the cluster."""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark

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
    def run_command(self, command, live_engine, *, apply: bool, findings: list | None = None,
                    **extra) -> tuple[int, list, str]:
        findings = findings or []
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
                mock.patch.object(spark, "start_findings", return_value=findings), \
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

    def test_the_plan_runs_the_preflight_and_names_the_fix(self) -> None:
        findings = [spark.Drift("dgx2: checkout a differs from deployment b"), spark.Warn("dgx3: 4 KiB kernel")]
        code, _, out = self.run_command(spark.command_cluster_start, None, apply=False, findings=findings)
        self.assertEqual(code, 0)
        self.assertIn("preflight: --apply would refuse:\n", out)
        self.assertIn("ERROR: dgx2: checkout a differs from deployment b", out)
        self.assertIn("WARN: dgx3: 4 KiB kernel", out)
        self.assertIn("fix: bin/spark --cluster-config config/cluster.json cluster sync --apply", out)

    def test_a_clean_preflight_says_so(self) -> None:
        code, _, out = self.run_command(spark.command_cluster_start, None, apply=False)
        self.assertEqual(code, 0)
        self.assertIn("preflight: clean", out)
        self.assertNotIn("fix:", out)

    def test_a_problem_sync_cannot_repair_names_no_sync(self) -> None:
        _, _, out = self.run_command(spark.command_cluster_start, None, apply=False,
                                     findings=["dgx2: checkout has uncommitted files"])
        self.assertIn("ERROR: dgx2: checkout has uncommitted files", out)
        self.assertNotIn("fix:", out)

    def test_apply_refuses_drift_with_the_same_fix(self) -> None:
        code, calls, out = self.run_command(spark.command_cluster_start, None, apply=True,
                                            findings=[spark.Drift("dgx2: image is unavailable: img-r6h")])
        self.assertEqual((code, calls), (1, []))
        self.assertIn("fix: bin/spark --cluster-config config/cluster.json cluster sync --apply", out)

    def test_an_unreachable_node_skips_the_preflight(self) -> None:
        args = argparse.Namespace(apply=False, cluster_config="config/cluster.json", replace=False)
        out = io.StringIO()
        with mock.patch.object(spark, "configuration", return_value=(CLUSTER, NODES3, {})), \
                mock.patch.object(spark, "expected_command", return_value=TP3), \
                mock.patch.object(spark, "live_engine", side_effect=RuntimeError("dgx2: unreachable")), \
                mock.patch.object(spark, "start_findings", side_effect=AssertionError("preflight reached")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(spark.command_cluster_start(args), 0)
        self.assertIn("--apply would refuse: dgx2: unreachable", out.getvalue())

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


class NodePreflightTest(unittest.TestCase):
    """The start problems cluster sync repairs are Drift, so the plan can name it."""

    CLUSTER = {"deployment": {"repository": "https://example.com/r", "path": "{home}/r"}, "host": {"home": "/h"},
               "container": {"name": "c", "image": "img", "expected_labels": {"vllm.tree": "t"}, "mounts": []}}

    def problems(self, node: str, *, at: str = "a", image: bool = True, tree: str = "t",
                 profile: bool = True, committed: bool = True) -> list[str]:
        doctors = []

        def run_ssh(_nodes, _node, *argv):
            if argv[:2] == ("test", "-f"):
                return subprocess.CompletedProcess(argv, 0 if profile else 1)
            if argv[:3] == ("docker", "image", "inspect"):
                if not image:
                    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Error: No such image: img")
                labels = spark.json.dumps([{"Config": {"Labels": {"vllm.tree": tree}}}])
                return subprocess.CompletedProcess(argv, 0, stdout=labels, stderr="")
            if argv[1:3] == ("--cluster-config", "experiments/x/cluster.json"):
                doctors.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with mock.patch.object(spark, "remote_repository_state", return_value=(at, "https://example.com/r", "")), \
                mock.patch.object(spark, "roce_gid_setting", return_value=(3, [])), \
                mock.patch.object(spark, "remote_shell", return_value=subprocess.CompletedProcess([], 0, "", "")), \
                mock.patch.object(spark, "roce_gid_findings", return_value=[]), \
                mock.patch.object(spark, "node_glp_problems", return_value=[]), \
                mock.patch.object(spark, "profile_committed", return_value=committed), \
                mock.patch.object(spark, "run_ssh", side_effect=run_ssh):
            found = spark.node_runtime_problems(self.CLUSTER, NODES3, spark.node_by_name(NODES3, node), "a",
                                                allow_existing=True, cluster_config="experiments/x/cluster.json")
        self.assertEqual(len(doctors), 1 if profile else 0)
        return found

    def drift(self, problems: list[str]) -> list[bool]:
        return [isinstance(problem, spark.Drift) for problem in problems]

    def test_a_clean_node_has_no_problem(self) -> None:
        self.assertEqual(self.problems("dgx2"), [])

    def test_a_checkout_behind_is_drift(self) -> None:
        problems = self.problems("dgx2", at="b")
        self.assertEqual((problems, self.drift(problems)), (["dgx2: checkout b differs from deployment a"], [True]))

    def test_a_workers_missing_or_stale_image_is_drift(self) -> None:
        for problems in (self.problems("dgx2", image=False), self.problems("dgx2", tree="old")):
            self.assertEqual(self.drift(problems), [True], problems)

    def test_the_heads_missing_or_stale_image_needs_a_build(self) -> None:
        missing = self.problems("dgx1", image=False)
        self.assertEqual(self.drift(missing), [False])
        self.assertIn("bin/spark --cluster-config experiments/x/cluster.json build image --apply", missing[0])
        self.assertEqual(self.drift(self.problems("dgx1", tree="old")), [False])

    def test_a_missing_profile_is_drift_only_when_the_commit_carries_it(self) -> None:
        behind = self.problems("dgx2", profile=False)
        self.assertEqual((behind, self.drift(behind)), (["dgx2: has no experiments/x/cluster.json"], [True]))
        uncommitted = self.problems("dgx2", profile=False, committed=False)
        self.assertEqual(self.drift(uncommitted), [False])
        self.assertIn("commit the profile, publish it and sync", uncommitted[0])


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


class SyncRevisionTest(unittest.TestCase):
    """A named revision is resolved after a fetch, so a commit merged since the last fetch
    (or origin/main) deploys instead of failing or detaching at a stale commit."""

    def test_fetch_before_resolving(self):
        calls = []
        args = argparse.Namespace(revision="origin/main", apply=False)
        revisions = {"origin/main": "abc", "HEAD": "def"}
        out = io.StringIO()
        with mock.patch.object(spark, "require_named_profile"), \
                mock.patch.object(spark, "configuration", return_value=({"x": 1}, {"nodes": [{"name": "n1"}]}, {})), \
                mock.patch.object(spark, "cluster_config_path", return_value=(None, "config/c.json")), \
                mock.patch.object(spark, "repository_path", return_value="/p"), \
                mock.patch.object(spark, "checked", side_effect=lambda command, **kw: calls.append(command[1])), \
                mock.patch.object(spark, "repository_revision",
                                  side_effect=lambda r="HEAD": calls.append(f"resolve {r}") or revisions[r]), \
                mock.patch.object(spark, "image_copies", side_effect=AssertionError("inspected the image")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(spark.command_cluster_sync(args), 0)
        self.assertEqual(calls, ["fetch", "resolve origin/main", "resolve HEAD"])
        # This checkout is not at the revision, so its profile may name another image.
        self.assertIn("image: then copy the one config/c.json names at abc", out.getvalue())


IMAGE_CLUSTER = {"container": {"name": "c", "image": "img", "expected_labels": {"vllm.tree": "t"}}}


class SyncImageTest(unittest.TestCase):
    """cluster sync gives every node the deployed profile's image, copied from this host."""

    def sync(self, *, at: str = "rev", problem: str | None = None, differ=(), failing: str | None = None):
        loads = []

        def load(_nodes, node, tag, layers):
            loads.append(node["name"])
            return f"{node['name']}: failed to load {tag}" if node["name"] == failing else None

        out = io.StringIO()
        with mock.patch.object(spark, "repository_revision", return_value=at), \
                mock.patch.object(spark, "configuration", return_value=(IMAGE_CLUSTER, NODES3, {})), \
                mock.patch.object(spark, "image_copies", return_value=(["sha256:l"], problem, list(differ))), \
                mock.patch.object(spark, "load_image", side_effect=load), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = spark.sync_image(argparse.Namespace(cluster_config="config/c.json"), "rev", "config/c.json")
        return code, loads, out.getvalue()

    def test_only_the_nodes_whose_copy_differs_load_it(self) -> None:
        code, loads, out = self.sync(differ=NODES3["nodes"][1:])
        self.assertEqual((code, loads), (0, ["dgx2", "dgx3"]))
        self.assertIn("every node holds img", out)

    def test_a_host_without_a_copy_to_give_warns_and_the_sync_stands(self) -> None:
        # A build may follow the sync (a lab image step builds on a synced node).
        code, loads, out = self.sync(problem="img is not built on this host")
        self.assertEqual((code, loads), (0, []))
        self.assertIn("WARN: image not copied: img is not built on this host", out)

    def test_a_checkout_off_the_deployed_commit_copies_nothing(self) -> None:
        code, loads, out = self.sync(at="other", differ=NODES3["nodes"][1:])
        self.assertEqual((code, loads), (0, []))
        self.assertIn("not at rev", out)

    def test_a_failed_load_fails_the_sync(self) -> None:
        code, loads, out = self.sync(differ=NODES3["nodes"][1:], failing="dgx2")
        self.assertEqual((code, loads), (1, ["dgx2"]))
        self.assertIn("ERROR: dgx2: failed to load img", out)

    def test_the_plan_names_the_copies(self) -> None:
        self.assertEqual(spark.image_plan("img", None, NODES3["nodes"][1:]), "image: copy img from this host to dgx2, dgx3")
        self.assertEqual(spark.image_plan("img", None, []), "image: every node has this host's img")
        self.assertEqual(spark.image_plan("img", "why", []), "image: not copied: why")


class LocalImageTest(unittest.TestCase):
    def inspect(self, code: int, image: dict | None = None, stderr: str = "", tag: str = "img"):
        result = subprocess.CompletedProcess([], code, stdout=spark.json.dumps([image]) if image else "", stderr=stderr)
        with mock.patch.object(spark.subprocess, "run", return_value=result):
            return spark.local_image(IMAGE_CLUSTER, tag, "config/c.json")

    def test_the_recipes_build_is_given(self) -> None:
        image = {"RootFS": {"Layers": ["sha256:a", "sha256:b"]}, "Config": {"Labels": {"vllm.tree": "t"}}}
        self.assertEqual(self.inspect(0, image), (["sha256:a", "sha256:b"], None))

    def test_an_unbuilt_image_names_the_build(self) -> None:
        _, problem = self.inspect(1, stderr="Error response from daemon: No such image: img")
        self.assertIn("bin/spark --cluster-config config/c.json build image --apply", problem)

    def test_a_stale_build_of_the_tag_is_not_given(self) -> None:
        image = {"RootFS": {"Layers": ["sha256:a"]}, "Config": {"Labels": {"vllm.tree": "old"}}}
        _, problem = self.inspect(0, image)
        self.assertIn("another build (label vllm.tree='old', the profile expects 't')", problem)
        # Another tag (build distribute --tag) is not the profile's image: its labels are not checked.
        self.assertEqual(self.inspect(0, image, tag="other"), (["sha256:a"], None))


class BuildDistributeTest(unittest.TestCase):
    def test_nodes_with_this_hosts_layers_are_skipped(self) -> None:
        loads = []
        args = argparse.Namespace(apply=True, cluster_config="config/cluster.json", tag=None)
        held = {"dgx1": ["sha256:l"], "dgx2": [], "dgx3": ["sha256:old"]}
        with mock.patch.object(spark, "configuration", return_value=(IMAGE_CLUSTER, NODES3, {})), \
                mock.patch.object(spark, "local_image", return_value=(["sha256:l"], None)), \
                mock.patch.object(spark, "image_layers", side_effect=lambda _n, node, _t: held[node["name"]]), \
                mock.patch.object(spark, "load_image", side_effect=lambda _n, node, *_: loads.append(node["name"])), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(spark.command_build_distribute(args), 0)
        self.assertEqual(loads, ["dgx2", "dgx3"])


class SyncProfileTest(unittest.TestCase):
    def test_a_profile_no_commit_carries_is_refused_before_any_checkout(self) -> None:
        cluster = {"deployment": {"repository": "https://example.com/r", "path": "/p"}, "host": {"home": "/h"},
                   **IMAGE_CLUSTER}
        args = argparse.Namespace(revision=None, apply=True, cluster_config="config/x.local.json")

        def run_ssh(_nodes, node, *argv):
            return subprocess.CompletedProcess(argv, 1 if node["name"] == "dgx2" else 0)

        published = subprocess.CompletedProcess([], 0, stdout="origin/main\n")
        with mock.patch.object(spark, "configuration", return_value=(cluster, NODES3, {})), \
                mock.patch.object(spark, "cluster_config_path", return_value=(None, "config/x.local.json")), \
                mock.patch.object(spark, "repository_revision", return_value="abc"), \
                mock.patch.object(spark, "require_local_deployment_state", return_value="abc"), \
                mock.patch.object(spark, "checked"), \
                mock.patch.object(spark.subprocess, "run", return_value=published), \
                mock.patch.object(spark, "profile_committed", return_value=False), \
                mock.patch.object(spark, "remote_repository_state", return_value=("old", "https://example.com/r", "")), \
                mock.patch.object(spark, "run_ssh", side_effect=run_ssh), \
                mock.patch.object(spark, "remote_shell", side_effect=AssertionError("fetched or checked out")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            spark.command_cluster_sync(args)
        self.assertIn("dgx2: has no config/x.local.json, and the deployed commit does not carry it", str(raised.exception))



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
                mock.patch.object(spark, "start_findings",
                                  side_effect=AssertionError("preflight reached")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                spark.command_cluster_start(args)
            except (SystemExit, AssertionError) as stop:
                return str(stop)
        return "started"

    def test_an_unqualified_shape_is_refused_unless_overridden(self) -> None:
        # The 4 KiB-page triangle was not benched with spark-ds41f-1025's image (the triangle was,
        # 2026-10-25, and is qualified below).
        for shape in ("tp3-triangle-4k",):
            unqualified = spark.cluster_recipe.resolve(spark.read_json("config/cluster.json"), shape)
            self.assertNotIn("benchmark_reference", unqualified)
            self.assertIn(f"no release benchmark qualifies this recipe's image on {shape}", self.start(unqualified))
            self.assertEqual(self.start(unqualified, unqualified_shape=True), "preflight reached")
        for shape in ("tp4-ring", "tp3-triangle"):
            qualified = spark.cluster_recipe.resolve(spark.read_json("config/cluster.json"), shape)
            self.assertEqual(self.start(qualified), "preflight reached", shape)
        # The frozen r6n recipe still starts the ring it was released on.
        frozen = spark.cluster_recipe.resolve(spark.read_json("experiments/2026-10-16-r6s/r6n/cluster.json"), "tp4-ring")
        self.assertEqual(self.start(frozen), "preflight reached")

    def test_a_concrete_profile_has_no_qualification_check(self) -> None:
        archived = spark.read_json("experiments/2026-10-14-one-recipe/tp3-r6h-production.json")
        self.assertNotIn("shape", archived)
        self.assertEqual(self.start(archived), "preflight reached")


if __name__ == "__main__":
    unittest.main()
