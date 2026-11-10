"""CPU-only contract and receipt verification tests for the TP3 6-channel bidirectional Ring NCCL profile."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))

from test_topology import spark, topology  # noqa: E402


EXPECTED_TP3_NCCL_ENV = {
    "NCCL_ALGO": "Ring",
    "NCCL_MIN_NCHANNELS": "6",
    "NCCL_MAX_NCHANNELS": "6",
    "NCCL_BUFFSIZE": "8388608",
    "NCCL_P2P_NET_CHUNKSIZE": "262144",
    "NCCL_SWITCHLESS_BIDIRECTIONAL": "1",
    "NCCL_MIN_TRAFFIC_PER_CHANNEL": "512",
    "NCCL_THREAD_THRESHOLDS": "-2 -2 -2 1 1 1",
}


class Tp3BidirectionalRingNcclTest(unittest.TestCase):
    def test_promoted_and_candidate_tp3_profiles_match_verified_6ch_bidirectional_ring(self):
        nodes = spark.read_json("config/nodes.example.json")
        configs = [
            ("config/cluster.json", "tp3-triangle"),
            ("config/cluster.json", "tp3-triangle-4k"),
            ("releases/spark-ds41f-1107/cluster.json", "tp3-triangle"),
            ("experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json", None),
            ("experiments/2026-11-08-tp3-bidirectional-ring-nccl/tp3.json", None),
        ]
        for path, shape in configs:
            with self.subTest(path=path, shape=shape):
                cluster = spark.read_cluster(path, shape) if shape else spark.read_cluster(path)
                cluster["distributed"]["master_addr"] = nodes["nodes"][0]["management_ip"]
                self.assertEqual(topology.problems(cluster, nodes), [])
                env = cluster["environment"]
                for key, expected in EXPECTED_TP3_NCCL_ENV.items():
                    self.assertEqual(env.get(key), expected, f"{path} ({shape}) mismatch for {key}")

    def test_control_and_p2p512k_ablation_overlays_validate_cleanly(self):
        nodes = spark.read_json("config/nodes.example.json")
        base_cluster = spark.read_cluster("experiments/2026-11-08-tp3-bidirectional-ring-nccl/base.json")
        base_cluster["distributed"]["master_addr"] = nodes["nodes"][0]["management_ip"]
        self.assertEqual(topology.problems(base_cluster, nodes), [])
        self.assertEqual(base_cluster["environment"]["NCCL_BUFFSIZE"], "1048576")
        self.assertEqual(base_cluster["environment"]["NCCL_MAX_NCHANNELS"], "8")
        for removed in (
            "NCCL_ALGO",
            "NCCL_MIN_NCHANNELS",
            "NCCL_P2P_NET_CHUNKSIZE",
            "NCCL_SWITCHLESS_BIDIRECTIONAL",
            "NCCL_MIN_TRAFFIC_PER_CHANNEL",
            "NCCL_THREAD_THRESHOLDS",
        ):
            self.assertNotIn(removed, base_cluster["environment"])

        p2p512k_cluster = spark.read_cluster("experiments/2026-11-08-tp3-bidirectional-ring-nccl/tp3-p2p512k.json")
        p2p512k_cluster["distributed"]["master_addr"] = nodes["nodes"][0]["management_ip"]
        self.assertEqual(topology.problems(p2p512k_cluster, nodes), [])
        self.assertEqual(p2p512k_cluster["environment"]["NCCL_P2P_NET_CHUNKSIZE"], "524288")

    def test_switchless_bidirectional_mode_2_is_rejected_for_6_channels_while_mode_1_is_valid(self):
        nodes = spark.read_json("config/nodes.example.json")
        cluster = spark.read_cluster("experiments/2026-11-08-tp3-bidirectional-ring-nccl/tp3.json")
        cluster["distributed"]["master_addr"] = nodes["nodes"][0]["management_ip"]
        self.assertEqual(topology.problems(cluster, nodes), [])

        cluster["environment"]["NCCL_SWITCHLESS_BIDIRECTIONAL"] = "2"
        errors = topology.problems(cluster, nodes)
        self.assertTrue(
            any("multiple of four" in err for err in errors),
            f"Expected mode 2 rejection for 6 channels, got {errors}",
        )

    def test_two_boot_receipts_and_results_summary_are_complete_and_bit_identical(self):
        results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["bit_identity_check_count"], 22)
        self.assertEqual(results["failed_bit_identity_checks"], [])

        inits = {(r["boot"], r["arm"]): r for r in results["nccl_inits"]}
        for boot in ("boot1", "boot2"):
            ctrl = inits[(boot, "control_c8_1m")]
            self.assertEqual(ctrl["coll_channels"], 8)
            self.assertEqual(ctrl["p2p_chunksize"], 131072)
            self.assertEqual(len(ctrl["rings"]), 8)
            self.assertTrue(all(r.endswith("2 -> 0 -> 1") for r in ctrl["rings"]))

            c256 = inits[(boot, "cand_c6_8m_bidir")]
            self.assertEqual(c256["coll_channels"], 6)
            self.assertEqual(c256["p2p_chunksize"], 524288)
            self.assertEqual(
                c256["rings"],
                [
                    "Ring 00 : 2 -> 0 -> 1",
                    "Ring 01 : 2 -> 0 -> 1",
                    "Ring 02 : 2 -> 0 -> 1",
                    "Ring 03 : 1 -> 0 -> 2",
                    "Ring 04 : 1 -> 0 -> 2",
                    "Ring 05 : 1 -> 0 -> 2",
                ],
            )

            c512 = inits[(boot, "cand_c6_8m_bidir_p2p512k")]
            self.assertEqual(c512["coll_channels"], 6)
            self.assertEqual(c512["p2p_chunksize"], 1048576)
            self.assertEqual(c512["rings"], c256["rings"])

        tokens_seen = {int(b["T"]) for b in results["bundle_totals"]}
        self.assertEqual(tokens_seen, {16, 256, 1024, 6144})
        self.assertEqual(len(results["comparisons"]), 48)


if __name__ == "__main__":
    unittest.main()
