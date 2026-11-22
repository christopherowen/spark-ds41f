"""Tests for config-driven pipeline parallelism (PP=2, TP=2..4)."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402
import topology  # noqa: E402
from spark_ds41f import cluster as spark_cluster  # noqa: E402
from spark_ds41f import config as spark_config  # noqa: E402
from spark_ds41f import doctor as spark_doctor  # noqa: E402


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _arg(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    return args[args.index(flag) + 1]


class PipelineParallelRecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = _load("config/cluster.json")

    def test_pp1_shapes_omit_pipeline_parallel_flag(self):
        resolved = recipe.resolve(self.recipe, "tp4-ring")
        self.assertNotIn("--pipeline-parallel-size", resolved["serve_args"])
        self.assertEqual(_arg(resolved["serve_args"], "--tensor-parallel-size"), "4")
        self.assertEqual(_arg(resolved["serve_args"], "--nnodes"), "4")

    def test_recipe_shape_resolves_pp2_tp2_tp3_tp4(self):
        cases = [
            ("pp2-tp2-ring", "tp4-ring", 4, 2, "oneshot-pp2-ring4", None),
            ("pp2-tp2-switched", "tp4-switched", 4, 2, "oneshot-switched", "switched"),
            ("pp2-tp3-switched", "tp6-switched", 6, 3, "oneshot-switched", "switched"),
            ("pp2-tp4-switched", "tp8-switched", 8, 4, "oneshot-switched", "switched"),
        ]
        for name, base_shape, nodes, expected_tp, transport, fabric_topo in cases:
            with self.subTest(shape=name):
                custom = copy.deepcopy(self.recipe)
                fabric = {"transport": transport}
                if fabric_topo:
                    fabric["topology"] = fabric_topo
                custom["shapes"][name] = {
                    "extends": base_shape,
                    "nodes": nodes,
                    "pipeline_parallel_size": 2,
                    "fabric": fabric,
                    "speculative": {
                        "enable_adaptive_verification": False,
                        "adaptive_verification_row_cost": None,
                    },
                    "environment": {"VLLM_SINK_REJECTION": "0"},
                }
                resolved = recipe.resolve(custom, name)
                args = resolved["serve_args"]
                self.assertEqual(_arg(args, "--tensor-parallel-size"), str(expected_tp))
                self.assertEqual(_arg(args, "--pipeline-parallel-size"), "2")
                self.assertEqual(_arg(args, "--nnodes"), str(nodes))
                spec = json.loads(_arg(args, "--speculative-config"))
                self.assertEqual(spec["draft_tensor_parallel_size"], expected_tp)
                self.assertIs(spec["enable_adaptive_verification"], False)
                self.assertNotIn("adaptive_verification_row_cost", spec)
                self.assertEqual(resolved["environment"]["VLLM_SINK_REJECTION"], "0")
                self.assertNotIn("VLLM_LATE_ADMISSION_LAYER", resolved["environment"])

    def test_non_divisible_nodes_and_pp_rejected(self):
        bad = copy.deepcopy(self.recipe)
        bad["shapes"]["bad-pp"] = {
            "extends": "tp5-switched",
            "nodes": 5,
            "pipeline_parallel_size": 2,
        }
        with self.assertRaises(SystemExit):
            recipe.resolve(bad, "bad-pp")

    def test_speculative_false_strips_flag_and_sizes_rows(self):
        custom = copy.deepcopy(self.recipe)
        custom["shapes"]["pp2-nospec"] = {
            "extends": "tp4-switched",
            "nodes": 4,
            "pipeline_parallel_size": 2,
            "speculative": False,
        }
        resolved = recipe.resolve(custom, "pp2-nospec")
        self.assertNotIn("--speculative-config", resolved["serve_args"])
        self.assertEqual(
            _arg(resolved["serve_args"], "--max-cudagraph-capture-size"),
            _arg(resolved["serve_args"], "--max-num-seqs"),
        )


class PipelineParallelTopologyAndOverlaysTest(unittest.TestCase):
    OVERLAY_CASES = [
        (
            "experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json",
            "config/examples/nodes-pp2-tp2.json",
            4,
            2,
            2,
            "pp2-tp2-ring",
        ),
        (
            "experiments/2026-11-20-pp2-tp2/pp2-tp2-socket.json",
            "config/examples/nodes-pp2-tp2.json",
            4,
            2,
            2,
            "pp2-tp2-socket",
        ),
        (
            "experiments/2026-11-20-pp2-tp2/pp2-tp2-switched.json",
            "config/examples/nodes-switched.json",
            4,
            2,
            2,
            "pp2-tp2-switched",
        ),
        (
            "experiments/2026-11-20-pp2-tp2/pp2-tp3-switched.json",
            "config/examples/nodes-switched6.json",
            6,
            3,
            2,
            "pp2-tp3-switched",
        ),
        (
            "experiments/2026-11-20-pp2-tp2/pp2-tp4-switched.json",
            "config/examples/nodes-switched8.json",
            8,
            4,
            2,
            "pp2-tp4-switched",
        ),
    ]

    def _aligned_nodes(self, cluster: dict, nodes_path: str) -> dict:
        nodes = _load(nodes_path)
        for node in nodes["nodes"]:
            if node.get("head"):
                node["management_ip"] = cluster["distributed"]["master_addr"]
        return nodes

    def test_all_pp2_experiment_overlays_validate(self):
        for cfg_path, nodes_path, count, tp, pp, expected_shape in self.OVERLAY_CASES:
            with self.subTest(config=cfg_path):
                cluster = spark_config.read_cluster(cfg_path)
                nodes = self._aligned_nodes(cluster, nodes_path)
                self.assertEqual(topology.problems(cluster, nodes), [])
                self.assertEqual(_arg(cluster["serve_args"], "--nnodes"), str(count))
                self.assertEqual(_arg(cluster["serve_args"], "--tensor-parallel-size"), str(tp))
                self.assertEqual(_arg(cluster["serve_args"], "--pipeline-parallel-size"), str(pp))
                spec = json.loads(_arg(cluster["serve_args"], "--speculative-config"))
                self.assertEqual(spec["draft_tensor_parallel_size"], tp)
                self.assertEqual(spark_doctor._cluster_shape(cluster), expected_shape)
                shape = spark_cluster.engine_shape(cluster["serve_args"])
                self.assertEqual(shape["pp"], str(pp))
                self.assertIn(f"PP{pp} TP{tp}", spark_cluster.engine_text(shape))

    def test_pp2_ring4_node_environment_splits_tp_and_pp_links(self):
        cluster = spark_config.read_cluster("experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json")
        nodes = self._aligned_nodes(cluster, "config/examples/nodes-pp2-tp2.json")
        expected_tp_peers = {
            0: ("1", ["rocep1s0f0", "roceP2p1s0f0"]),
            1: ("0", ["rocep1s0f1", "roceP2p1s0f1"]),
            2: ("1", ["rocep1s0f1", "roceP2p1s0f1"]),
            3: ("0", ["rocep1s0f0", "roceP2p1s0f0"]),
        }
        for node in nodes["nodes"]:
            rank = node["rank"]
            env = topology.node_environment(cluster, node)
            self.assertEqual(
                env["NCCL_IB_HCA"],
                "=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1",
            )
            self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "direct")
            local_peer, tp_hcas = expected_tp_peers[rank]
            self.assertEqual(
                json.loads(env["SPARKNET_ROCE_PEER_HCAS"]),
                {local_peer: tp_hcas},
            )
            self.assertEqual(env["NCCL_IB_SUBNET_AWARE_ROUTING"], "1")
            self.assertEqual(env["NCCL_RUNTIME_CONNECT"], "1")
            self.assertNotIn("NCCL_SWITCHLESS_BIDIRECTIONAL", env)

    def test_pp2_ring4_node_environment_without_gid_pin_sets_tp_subnets_only(self):
        cluster = spark_config.read_cluster("experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json")
        cluster["environment"].pop("NCCL_IB_GID_INDEX", None)
        nodes = self._aligned_nodes(cluster, "config/examples/nodes-pp2-tp2.json")
        env0 = topology.node_environment(cluster, nodes["nodes"][0])
        self.assertEqual(
            json.loads(env0["SPARKNET_ROCE_SUBNETS"]),
            {"rocep1s0f0": "10.40.0.0/24", "roceP2p1s0f0": "10.41.0.0/24"},
        )

    def test_pp2_ring4_node_map_derives_valid_layout_from_ring4(self):
        ring4 = _load("config/examples/nodes-ring4.json")
        derived = topology.pp2_ring4_node_map(ring4)
        self.assertEqual(
            [(n["name"], n["rank"]) for n in derived["nodes"]],
            [("dgx1", 0), ("dgx2", 1), ("dgx4", 2), ("dgx3", 3)],
        )
        cluster = spark_config.read_cluster("experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json")
        derived["nodes"][0]["management_ip"] = cluster["distributed"]["master_addr"]
        self.assertEqual(topology.problems(cluster, derived), [])

    def test_topology_rejects_invalid_pp_configurations(self):
        # PP=2 on oneshot-ring4 is invalid (must use oneshot-pp2-ring4).
        ring = spark_config.read_cluster("experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json")
        ring_nodes = self._aligned_nodes(ring, "config/examples/nodes-pp2-tp2.json")

        bad_bidir = copy.deepcopy(ring)
        bad_bidir["environment"]["NCCL_SWITCHLESS_BIDIRECTIONAL"] = "2"
        self.assertTrue(
            any("NCCL_SWITCHLESS_BIDIRECTIONAL" in err for err in topology.problems(bad_bidir, ring_nodes))
        )

        bad_draft = copy.deepcopy(ring)
        idx = bad_draft["serve_args"].index("--speculative-config") + 1
        spec = json.loads(bad_draft["serve_args"][idx])
        spec["draft_tensor_parallel_size"] = 4
        bad_draft["serve_args"][idx] = json.dumps(spec)
        self.assertTrue(
            any("draft_tensor_parallel_size" in err for err in topology.problems(bad_draft, ring_nodes))
        )

        bad_subnet = copy.deepcopy(ring_nodes)
        bad_subnet["nodes"][2]["roce_subnets"]["rocep1s0f0"] = "10.99.1.0/24"
        self.assertTrue(
            any("mismatched roce_subnets" in err for err in topology.problems(ring, bad_subnet))
        )

    def test_candidate_69_pp2_shapes_resolve_and_pass_doctor(self):
        cases = [
            ("pp2-tp2-ring", 4, 2, "oneshot-pp2-ring4", "config/examples/nodes-pp2-tp2.json"),
            ("pp2-tp2-switched", 4, 2, "oneshot-switched", "config/examples/nodes-switched.json"),
            ("pp2-tp3-switched", 6, 3, "oneshot-switched", "config/examples/nodes-switched6.json"),
            ("pp2-tp4-switched", 8, 4, "oneshot-switched", "config/examples/nodes-switched8.json"),
        ]
        c69_lock = spark_config.read_json(
            "experiments/2026-10-26-spark-ds41f-candidate/candidate-69/upstreams.lock.json"
        )
        for shape, expected_nodes, expected_tp, expected_transport, example_nodes in cases:
            with self.subTest(source="candidate-69", shape=shape):
                c69 = spark_config.read_cluster(
                    f"experiments/2026-10-26-spark-ds41f-candidate/candidate-69/{shape}.json"
                )
                c69_dev = spark_config.read_cluster(
                    f"experiments/2026-10-26-spark-ds41f-candidate/candidate-69/{shape}-dev.json"
                )
                nodes = self._aligned_nodes(c69, example_nodes)
                self.assertEqual(_arg(c69["serve_args"], "--nnodes"), str(expected_nodes))
                self.assertEqual(_arg(c69["serve_args"], "--tensor-parallel-size"), str(expected_tp))
                self.assertEqual(_arg(c69["serve_args"], "--pipeline-parallel-size"), "2")
                self.assertEqual(c69["fabric"]["transport"], expected_transport)
                self.assertNotIn("VLLM_LATE_ADMISSION_LAYER", c69["environment"])
                if expected_transport == "oneshot-pp2-ring4":
                    self.assertNotIn("NCCL_SWITCHLESS_BIDIRECTIONAL", c69["environment"])
                self.assertEqual(c69["container"]["image"], "spark-ds41f:candidate-69")
                self.assertEqual(topology.problems(c69, nodes), [])
                self.assertEqual(spark_doctor._cluster_shape(c69), shape)
                self.assertEqual(c69_dev["environment"].get("VLLM_SERVER_DEV_MODE"), "1")
                self.assertEqual(spark_doctor.qualified_recipe_problems(c69, nodes, c69_lock), [])


if __name__ == "__main__":
    unittest.main()
