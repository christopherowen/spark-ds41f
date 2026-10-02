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
loader = importlib.machinery.SourceFileLoader("spark3_topology_test", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
topology = spark3.topology


class TopologyTest(unittest.TestCase):
    def setUp(self):
        self.base = spark3.read_json("config/cluster.json")
        self.three = spark3.read_json("config/nodes.json")
        self.four = spark3.read_json("config/examples/nodes-ring4.json")
        self.ring = topology.candidate(self.base, self.four, "config/examples/nodes-ring4.json")

    def test_promoted_environment_and_commands_are_unchanged(self):
        self.assertEqual(topology.problems(self.base, self.three), [])
        for node in self.three["nodes"]:
            expected = {k: str(v) for k, v in self.base["environment"].items()}
            expected.update(VLLM_HOST_IP=node["management_ip"],
                            B12X_ROCE_PEER_HCAS=json.dumps(node["roce_peer_hcas"], separators=(",", ":")))
            self.assertEqual(spark3.expected_environment(self.base, node), expected)
            command = spark3.expected_command(self.base, node)
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
        command = spark3.rendered_docker_command(self.ring, node)
        self.assertIn("--headless", command)
        self.assertEqual(command[command.index("--node-rank") + 1], "3")
        env = spark3.expected_environment(self.ring, node)
        self.assertNotIn("B12X_ROCE_PEER_HCAS", env)
        self.assertEqual(env["VLLM_ENABLE_ROCE_ALLREDUCE"], "0")
        self.assertTrue(env["NCCL_IB_HCA"].startswith("="))

    def test_cli_accepts_a_fourth_node_and_validates_against_selected_map(self):
        args = spark3.parser().parse_args(["--cluster-config", "ring4.json", "render", "dgx4"])
        self.assertEqual(args.node, "dgx4")
        with mock.patch.object(spark3, "configuration", return_value=(self.ring, self.four, {})), mock.patch("builtins.print") as output:
            self.assertEqual(args.func(args), 0)
            self.assertIn("--node-rank 3", str(output.call_args))
        with self.assertRaises(SystemExit):
            spark3.node_by_name(self.three, "dgx4")

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
        index, _ = spark3.roce_gid_setting(self.ring, node)
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
        command = spark3.collective_probe_command(self.ring, self.four, self.four["nodes"][3], 29999)
        self.assertIn("180s", command)
        self.assertIn("--memory=4g", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn("--entrypoint=/usr/bin/timeout", command)
        self.assertEqual(command[command.index("--world-size") + 1], "4")

    def test_nodes_path_round_trip_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.json").write_text(json.dumps(self.four))
            candidate = copy.deepcopy(self.ring)
            candidate["nodes_config"] = "nodes.json"
            (root / "cluster.json").write_text(json.dumps(candidate))
            (root / "upstreams.lock.json").write_text("{}")
            with mock.patch.object(spark3, "ROOT", root):
                _, nodes, _ = spark3.configuration(argparse.Namespace(cluster_config="cluster.json"))
                self.assertEqual(nodes, self.four)
                for path in ("../escape.json", "/tmp/escape.json"):
                    with self.assertRaises(SystemExit):
                        spark3.repository_config_path(path)

    def test_create_never_overwrites_an_existing_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.json").write_text(json.dumps(self.four))
            (root / "candidate.json").write_text("preserve me")
            with mock.patch.object(spark3, "ROOT", root), mock.patch.object(spark3, "configuration", return_value=(self.base, self.three, {})):
                with self.assertRaises(FileExistsError):
                    spark3.command_topology_create(argparse.Namespace(nodes_config="nodes.json", output="candidate.json"))
            self.assertEqual((root / "candidate.json").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
