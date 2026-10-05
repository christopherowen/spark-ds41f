"""Topology mistakes must fail before contacting a node or opening a QP."""

import argparse
import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_topology_test", str(ROOT / "bin/spark"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)
topology = spark.topology


# The B12X TP3 configuration the topology tooling generates from: r5p's, kept
# with r5p's lock when r6 promoted the TileLang family.
B12X_BASE = "experiments/2026-10-05-tilelang-r6/r5p/cluster-64k.json"


class TopologyTest(unittest.TestCase):
    def setUp(self):
        self.base = spark.read_json(B12X_BASE)
        self.three = spark.read_json("config/nodes.json")
        # The site map (or CI's example map) names the API head.
        head = next(node for node in self.three["nodes"] if node.get("head"))
        self.base["distributed"]["master_addr"] = head["management_ip"]
        self.four = spark.read_json("config/examples/nodes-ring4.json")
        self.ring = topology.candidate(self.base, self.four, "config/examples/nodes-ring4.json")

    def test_promoted_environment_and_commands_are_unchanged(self):
        self.assertEqual(topology.problems(self.base, self.three), [])
        for node in self.three["nodes"]:
            expected = {k: str(v) for k, v in self.base["environment"].items()}
            expected.update(VLLM_HOST_IP=node["management_ip"],
                            B12X_ROCE_PEER_HCAS=json.dumps(node["roce_peer_hcas"], separators=(",", ":")))
            self.assertEqual(spark.expected_environment(self.base, node), expected)
            command = spark.expected_command(self.base, node)
            self.assertEqual(command[:len(self.base["serve_args"])], self.base["serve_args"])

    def test_four_node_candidate_has_consistent_target_and_draft_sizes(self):
        original = copy.deepcopy(self.base)
        self.assertEqual(topology.problems(self.ring, self.four), [])
        for flag in ("--tensor-parallel-size", "--nnodes"):
            self.assertEqual(topology.argument(self.ring, flag), "4")
        self.assertEqual(json.loads(topology.argument(self.ring, "--speculative-config"))["draft_tensor_parallel_size"], 4)
        self.assertEqual(self.ring["container"], self.base["container"])
        self.assertEqual(self.base, original)
        self.assertFalse(self.ring["deployment"]["launch_enabled"])
        self.assertNotIn("branch", self.ring["deployment"])

    def test_fourth_node_is_rendered_headless_without_rocenante_peer_map(self):
        node = self.four["nodes"][3]
        command = spark.rendered_docker_command(self.ring, node)
        self.assertIn("--headless", command)
        self.assertEqual(command[command.index("--node-rank") + 1], "3")
        env = spark.expected_environment(self.ring, node)
        self.assertNotIn("B12X_ROCE_PEER_HCAS", env)
        self.assertEqual(env["VLLM_ENABLE_ROCE_ALLREDUCE"], "0")
        self.assertTrue(env["NCCL_IB_HCA"].startswith("="))

    def test_cli_accepts_a_fourth_node_and_validates_against_selected_map(self):
        args = spark.parser().parse_args(["--cluster-config", "ring4.json", "render", "dgx4"])
        self.assertEqual(args.node, "dgx4")
        with mock.patch.object(spark, "configuration", return_value=(self.ring, self.four, {})), mock.patch("builtins.print") as output:
            self.assertEqual(args.func(args), 0)
            self.assertIn("--node-rank 3", str(output.call_args))
        with self.assertRaises(SystemExit):
            spark.node_by_name(self.three, "dgx4")

    def test_three_node_candidate_stays_on_existing_fast_path(self):
        candidate = topology.candidate(self.base, self.three, "config/nodes.json")
        self.assertEqual(topology.problems(candidate, self.three), [])
        self.assertEqual(candidate["environment"], self.base["environment"])
        self.assertEqual(candidate["serve_args"], self.base["serve_args"])

    def test_four_nodes_cannot_use_direct_peer_transport(self):
        self.ring.pop("fabric")
        self.assertTrue(any("requires nccl-ring" in p for p in topology.problems(self.ring, self.four)))

    def test_ring_policy_rejects_each_unsafe_override(self):
        for key in topology.RING_ENV:
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.ring)
                candidate["environment"][key] = "invalid"
                self.assertTrue(any(key in p for p in topology.problems(candidate, self.four)))

    def test_stale_three_node_argument_and_draft_config_rejected(self):
        for flag in ("--tensor-parallel-size", "--nnodes"):
            candidate = copy.deepcopy(self.ring)
            topology.set_argument(candidate, flag, "3")
            self.assertTrue(any(flag in p for p in topology.problems(candidate, self.four)))
        topology.set_argument(self.ring, "--speculative-config", '{"draft_tensor_parallel_size":3}')
        self.assertTrue(any("draft_tensor" in p for p in topology.problems(self.ring, self.four)))

    def test_qualified_ring_channels_preserve_neighbor_constraints(self):
        for channels in ("1", "2", "4", "8"):
            candidate = copy.deepcopy(self.ring)
            for key in ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"):
                candidate["environment"][key] = channels
            self.assertEqual(topology.problems(candidate, self.four), [])
            env = topology.node_environment(candidate, self.four["nodes"][0])
            self.assertEqual(env["NCCL_MAX_NCHANNELS"], channels)
            candidate["environment"]["NCCL_ALGO"] = "Tree"
            self.assertTrue(any("NCCL_ALGO" in p for p in topology.problems(candidate, self.four)))
        for lower, upper in (("1", "4"), ("3", "3"), ("0", "0"), ("16", "16")):
            candidate = copy.deepcopy(self.ring)
            candidate["environment"].update(NCCL_MIN_NCHANNELS=lower, NCCL_MAX_NCHANNELS=upper)
            self.assertTrue(any("NCCL_MIN_NCHANNELS" in p for p in topology.problems(candidate, self.four)))

    def test_ep_and_multiple_parallel_groups_rejected(self):
        self.ring["serve_args"].append("--enable-expert-parallel")
        topology.set_argument(self.ring, "--pipeline-parallel-size", "2")
        errors = topology.problems(self.ring, self.four)
        self.assertTrue(any("all-to-all" in p for p in errors))
        self.assertTrue(any("pipeline" in p for p in errors))

    def test_ring_requires_neighbours_in_rank_order(self):
        node = self.four["nodes"][0]
        node["roce_peer_hcas"]["2"] = node["roce_peer_hcas"].pop("3")
        self.assertTrue(any("cable/rank order" in p for p in topology.problems(self.ring, self.four)))

    def test_ring_rejects_reused_interface_and_asymmetric_stripes(self):
        node = self.four["nodes"][0]
        node["roce_peer_hcas"]["1"] = ["rocep1s0f1"]
        errors = topology.problems(self.ring, self.four)
        self.assertTrue(any("distinct local HCAs" in p for p in errors))
        self.assertTrue(any("reciprocal" in p for p in errors))

    def test_ring_rejects_missing_or_shared_cable_subnets(self):
        self.four["nodes"][0]["roce_subnets"].pop("rocep1s0f0")
        errors = topology.problems(self.ring, self.four)
        self.assertTrue(any("roce_subnets" in p for p in errors))
        self.assertTrue(any("exactly the two" in p for p in errors))

    def test_duplicate_ranks_and_missing_head_rejected(self):
        self.four["nodes"][3]["rank"] = 2
        self.assertTrue(any("contiguous" in p for p in topology.problems(self.ring, self.four)))
        self.three["nodes"][0]["head"] = False
        self.assertTrue(any("API head" in p for p in topology.problems(self.base, self.three)))

    def test_live_gid_matches_declared_cable_and_per_node_index(self):
        node = self.four["nodes"][0]
        node["roce_gid_index"] = 4
        index, _ = spark.roce_gid_setting(self.ring, node)
        self.assertEqual(index, 4)
        lines = []
        for hca, subnet in node["roce_subnets"].items():
            address = subnet.replace(".0/24", ".1")
            lines.append(f"{hca} 4 RoCEv2 ::ffff:{address} en0")
        output = "\n".join(lines)
        self.assertEqual(topology.gid_subnet_problems(node, index, output), [])
        self.assertTrue(topology.gid_subnet_problems(node, index, output.replace("10.40.0.1", "10.40.2.1")))
        self.assertTrue(topology.gid_subnet_problems(node, 3, output))

    def test_probe_is_bounded_and_does_not_share_serving_ipc(self):
        command = spark.collective_probe_command(self.ring, self.four, self.four["nodes"][3], 29999)
        self.assertIn("600s", command)
        self.assertIn("--memory=12g", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn("--entrypoint=/usr/bin/timeout", command)
        self.assertEqual(command[command.index("--world-size") + 1], "4")

    def test_nodes_path_round_trip_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.json").write_text(json.dumps(self.four))
            candidate = copy.deepcopy(self.ring)
            candidate["nodes_config"] = "nodes.json"
            candidate.pop("upstreams_config", None)  # the temporary root holds its own lock
            (root / "cluster.json").write_text(json.dumps(candidate))
            (root / "upstreams.lock.json").write_text("{}")
            with mock.patch.object(spark, "ROOT", root):
                _, nodes, _ = spark.configuration(argparse.Namespace(cluster_config="cluster.json"))
                self.assertEqual(nodes, self.four)
                for path in ("../escape.json", "/tmp/escape.json"):
                    with self.assertRaises(SystemExit):
                        spark.repository_config_path(path)

    def test_rocenante_ring4_candidate_and_source_lock(self):
        args = argparse.Namespace(cluster_config="experiments/2026-10-02-rocenante-ring4/cluster.json")
        cluster, nodes, lock = spark.configuration(args)
        self.assertEqual(topology.problems(cluster, nodes), [])
        self.assertEqual(spark.local_doctor(cluster, nodes, lock), [])
        self.assertIn("2026-10-02-rocenante-ring4", lock["source_manifest"])
        self.assertFalse(cluster["deployment"]["launch_enabled"])
        self.assertNotEqual(cluster["container"]["image"], self.base["container"]["image"])
        for node in nodes["nodes"]:
            env = spark.expected_environment(cluster, node)
            self.assertEqual(env["VLLM_ENABLE_ROCE_ALLREDUCE"], "1")
            self.assertEqual(env["B12X_ROCE_TOPOLOGY"], "ring4")
            self.assertEqual(set(json.loads(env["B12X_ROCE_PEER_HCAS"])),
                             {str((node["rank"]-1)%4), str((node["rank"]+1)%4)})
            self.assertNotIn("--disable-custom-all-reduce", spark.expected_command(cluster, node))

    def test_relay_stripe_order_must_match_the_remote_cable(self):
        relay = spark.read_json("experiments/2026-10-02-rocenante-ring4/cluster.json")
        self.four["nodes"][0]["roce_peer_hcas"]["1"].reverse()
        self.assertTrue(any("same stripe position" in p for p in topology.problems(relay, self.four)))
        # NCCL selects by subnet rather than route-list position.
        self.assertEqual(topology.problems(self.ring, self.four), [])

    def test_relay_base_can_generate_nccl_control(self):
        relay = spark.read_json("experiments/2026-10-02-rocenante-ring4/cluster.json")
        control = topology.candidate(relay, self.four, "config/examples/nodes-ring4.json")
        self.assertEqual(topology.problems(control, self.four), [])
        self.assertEqual(control["container"], relay["container"])
        self.assertEqual(topology.transport(control), "nccl-ring")
        self.assertNotIn("B12X_ROCE_TOPOLOGY", control["environment"])

    def test_rocenante_ring4_rejects_old_image_or_lock(self):
        args = argparse.Namespace(cluster_config="experiments/2026-10-02-rocenante-ring4/cluster.json")
        cluster, nodes, lock = spark.configuration(args)
        cluster["container"]["expected_labels"] = self.base["container"]["expected_labels"]
        self.assertTrue(any("image tree" in p for p in spark.local_doctor(cluster, nodes, lock)))
        # r5o's source manifest predates the ring transport patch.
        old_lock = spark.read_json("upstreams.lock.json")
        old_lock["source_manifest"] = "manifests/sources/2026-09-30-r5o-candidate-source.json"
        self.assertTrue(any("transport patch" in p for p in spark.local_doctor(cluster, nodes, old_lock)))

    def test_rocenante_ring4_requires_matching_mode_and_backend(self):
        cluster = spark.read_json("experiments/2026-10-02-rocenante-ring4/cluster.json")
        for key in ("VLLM_ENABLE_ROCE_ALLREDUCE", "B12X_ROCE_TOPOLOGY"):
            changed = copy.deepcopy(cluster)
            changed["environment"][key] = "invalid"
            self.assertTrue(any(key in p for p in topology.problems(changed, self.four)))
        cluster["serve_args"].append("--disable-custom-all-reduce")
        self.assertTrue(any("custom all-reduce enabled" in p for p in topology.problems(cluster, self.four)))
        self.base["environment"]["B12X_ROCE_TOPOLOGY"] = "ring4"
        self.assertTrue(any("B12X_ROCE_TOPOLOGY" in p for p in topology.problems(self.base, self.three)))

    def sparknet(self):
        """The sparknet candidate on the example ring map, with its lock."""
        experiment = "experiments/2026-10-04-tilelang-1m"
        cluster = spark.read_json(f"{experiment}/candidate.json")
        cluster["distributed"]["master_addr"] = next(n for n in self.four["nodes"] if n.get("head"))["management_ip"]
        return cluster, spark.read_json(f"{experiment}/upstreams.lock.json")

    def test_oneshot_ring4_runs_sparknet_under_its_own_names(self):
        cluster, lock = self.sparknet()
        self.assertEqual(topology.transport(cluster), "oneshot-ring4")
        self.assertEqual(topology.problems(cluster, self.four), [])
        self.assertEqual([p for p in spark.local_doctor(cluster, self.four, lock) if not isinstance(p, spark.Warn)], [])
        for node in self.four["nodes"]:
            env = spark.expected_environment(cluster, node)
            self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "ring4")
            self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])),
                             {str((node["rank"] - 1) % 4), str((node["rank"] + 1) % 4)})
            self.assertEqual(env["SPARKNET_ROCE_GID_INDEX"], str(node.get("roce_gid_index", 3)))
            self.assertFalse([k for k in env if k.startswith("B12X_ROCE_") or k in topology.B12X_VLLM_SETTINGS])
            self.assertNotIn("--disable-custom-all-reduce", spark.expected_command(cluster, node))
        self.assertEqual(spark.roce_gid_setting(cluster, self.four["nodes"][1])[0],
                         self.four["nodes"][1].get("roce_gid_index", 3))

    def test_oneshot_refuses_b12x_settings_and_mismatched_images(self):
        cluster, lock = self.sparknet()
        for key, value in (("VLLM_ENABLE_ROCE_ALLREDUCE", "1"), ("B12X_ROCE_TOPOLOGY", "ring4"),
                           ("VLLM_ROCE_ALLREDUCE_MAX_SIZE", "2MB")):
            changed = copy.deepcopy(cluster)
            changed["environment"][key] = value
            self.assertTrue(any(key in p for p in topology.problems(changed, self.four)), key)
        changed = copy.deepcopy(cluster)
        changed["environment"]["SPARKNET_ROCE_TOPOLOGY"] = "direct"
        self.assertTrue(any("SPARKNET_ROCE_TOPOLOGY" in p for p in topology.problems(changed, self.four)))
        changed = copy.deepcopy(cluster)
        changed["serve_args"].append("--disable-custom-all-reduce")
        self.assertTrue(any("custom all-reduce enabled" in p for p in topology.problems(changed, self.four)))
        relay = spark.read_json("experiments/2026-10-02-rocenante-ring4/cluster.json")
        relay["environment"]["SPARKNET_ROCE_SPIN_LIMIT"] = "1"
        self.assertTrue(any("does not run sparknet" in p for p in topology.problems(relay, self.four)))
        # A sparknet image serves only oneshot-*; oneshot-* needs that image.
        b12x = copy.deepcopy(cluster)
        b12x["fabric"]["transport"] = "rocenante-ring4"
        self.assertTrue(any("serves sparknet's oneshot-* transports" in str(p)
                            for p in spark.local_doctor(b12x, self.four, lock)))
        older = spark.read_json("experiments/2026-10-04-streamed-embeddings/upstreams.lock.json")
        self.assertTrue(any("requires an image with" in str(p) for p in spark.local_doctor(cluster, self.four, older)))

    def test_create_never_overwrites_an_existing_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.json").write_text(json.dumps(self.four))
            (root / "candidate.json").write_text("preserve me")
            with mock.patch.object(spark, "ROOT", root), mock.patch.object(spark, "configuration", return_value=(self.base, self.three, {})):
                with self.assertRaises(FileExistsError):
                    spark.command_topology_create(argparse.Namespace(nodes_config="nodes.json", output="candidate.json"))
            self.assertEqual((root / "candidate.json").read_text(), "preserve me")

    def test_former_repository_url_names_the_renamed_repository(self):
        current = "https://github.com/christopherowen/spark-ds41f.git"
        self.assertEqual(self.base["deployment"]["repository"], current)
        for former in ("https://github.com/christopherowen/spark3-vllm-ds41f",
                       "https://github.com/christopherowen/spark3-vllm-ds41f.git/"):
            self.assertEqual(spark.normalized_repository_url(former),
                             spark.normalized_repository_url(current))
        self.assertNotEqual(spark.normalized_repository_url("https://github.com/example/spark-ds41f"),
                            spark.normalized_repository_url(current))


if __name__ == "__main__":
    unittest.main()
