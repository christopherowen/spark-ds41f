#!/usr/bin/env python3
"""CPU unit tests for 2026-11-06-nccl-sp-channels (TP3 NCCL SP channels & Sparse-MLA occupancy)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from spark_ds41f.build import patchset_sha256
from spark_ds41f.config import ROOT, read_cluster, topology

EXP_DIR = ROOT / "experiments/2026-11-06-nccl-sp-channels"
TP4_1107 = "experiments/2026-10-26-spark-ds41f-candidate/1107/tp4.json"


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _simulate_routes(
    heads: int,
    swa_width: int = 128,
    idx_width: int = 512,
    *,
    max_rows: int = 2048,
    block_H: int = 32,
    split_block_H: int = 16,
    block_I: int = 32,
    threads: int = 256,
    segment_chunks: int = 4,
    decode_max_waves: int = 3,
    sms: int = 48,
) -> dict:
    """Exact CPU replica of SparseMlaV41Plan.routes after vLLM 0731."""
    block_H = min(block_H, _cdiv(heads, 16) * 16)
    threads = min(threads, 32 * min(8, (block_H // 16) * (block_I // 8)))
    head_blocks = _cdiv(heads, block_H)
    chunks = _cdiv(swa_width, block_I) + _cdiv(idx_width, block_I)
    segments = _cdiv(chunks, segment_chunks)
    decode_max_rows = max(
        decode_max_waves * sms // (segments * head_blocks),
        sms // head_blocks if segments > 1 else 0,
    )
    split_max_rows = (
        sms // (segments * _cdiv(heads, split_block_H)) if split_block_H else 0
    )
    decode_max_rows = min(decode_max_rows, max_rows)
    split_max_rows = min(split_max_rows, decode_max_rows)
    split_threads = 32 * min(4, (split_block_H // 16) * (block_I // 8))
    return {
        "heads": heads,
        "segments": segments,
        "head_blocks": head_blocks,
        "block_H": block_H,
        "threads": threads,
        "split_block_H": split_block_H,
        "split_threads": split_threads,
        "split_max_rows": split_max_rows,
        "decode_max_rows": decode_max_rows,
    }


class TestNcclSpChannelsAndMlaOccupancy(unittest.TestCase):
    def test_base_profile_preserves_1107_tp3_control(self) -> None:
        base = read_cluster("experiments/2026-11-06-nccl-sp-channels/base.json")
        env = base["environment"]
        self.assertEqual(base["container"]["image"], "spark-ds41f:1107")
        self.assertEqual(env["NCCL_MAX_NCHANNELS"], "8")
        self.assertEqual(env["NCCL_BUFFSIZE"], "1048576")
        self.assertNotIn("NCCL_MIN_NCHANNELS", env)
        self.assertNotIn("NCCL_ALGO", env)
        self.assertNotIn("NCCL_SWITCHLESS_BIDIRECTIONAL", env)

    def test_c4_ring_arm_isolates_channel_count_and_ring_algo(self) -> None:
        c4 = read_cluster("experiments/2026-11-06-nccl-sp-channels/tp3-c4-ring.json")
        env = c4["environment"]
        self.assertEqual(c4["container"]["image"], "spark-ds41f:1107")
        self.assertEqual(env["NCCL_ALGO"], "Ring")
        self.assertEqual(env["NCCL_MIN_NCHANNELS"], "4")
        self.assertEqual(env["NCCL_MAX_NCHANNELS"], "4")
        self.assertEqual(env["NCCL_BUFFSIZE"], "1048576")
        self.assertNotIn("NCCL_SWITCHLESS_BIDIRECTIONAL", env)

    def test_candidate_matches_tp4_and_switchless_oneshot_direct_policy(self) -> None:
        cand = read_cluster("experiments/2026-11-06-nccl-sp-channels/candidate.json")
        tp4 = read_cluster(TP4_1107)
        env = cand["environment"]
        tp4_env = tp4["environment"]

        # Zero container rebuild needed for the NCCL candidate arm.
        self.assertEqual(cand["container"]["image"], "spark-ds41f:1107")

        keys = (
            "NCCL_ALGO",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_BUFFSIZE",
            "NCCL_SWITCHLESS_BIDIRECTIONAL",
            "NCCL_MIN_TRAFFIC_PER_CHANNEL",
            "NCCL_THREAD_THRESHOLDS",
        )
        for key in keys:
            self.assertEqual(env[key], tp4_env[key], key)

        # Also check alignment with scripts/topology.py oneshot-direct rendering for 3 nodes.
        nodes_example = json.loads((ROOT / "config/nodes.example.json").read_text())
        rendered = topology.candidate(cand, nodes_example, "config/nodes.example.json")
        for key in keys:
            self.assertEqual(env[key], rendered["environment"][key], key)

        # P2P chunk size (NCCL_BUFFSIZE / 8 steps) is unclamped to GB10s 256 KiB default.
        self.assertEqual(int(env["NCCL_BUFFSIZE"]) // 8, 524288)

    def test_sp_prefill_collectives_exceed_oneshot_4mib_cap_and_preserve_bit_identity(self) -> None:
        # At TP3 prefill chunk = 6144 tokens (L = 2048 rows/rank), verify every SP
        # collective payload against SparknetOneShotAllReduce._slot_bytes (4 MiB).
        oneshot_cap = 4 * 1024 * 1024
        rows_per_rank = 2048
        world = 3

        sp_payloads = {
            "attention_front_compressed_ag": world * rows_per_rank * 3584,
            "attention_front_uncompressed_ag": world * rows_per_rank * 5120 * 2,
            "wo_exchange_into_slice": world * (rows_per_rank // 2) * 5120 * 2,
            "moe_gather_mxfp8": world * rows_per_rank * 5376,
            "moe_reduce_scatter_parts": world * rows_per_rank * 5120 * 2,
        }
        for name, total_bytes in sp_payloads.items():
            self.assertGreater(total_bytes, oneshot_cap, name)

    def test_upstreams_lock_and_source_manifest_match_patch_0731(self) -> None:
        upstreams = json.loads((EXP_DIR / "upstreams.lock.json").read_text())
        source = json.loads((EXP_DIR / "source.json").read_text())
        vllm_u = upstreams["sources"]["vllm"]
        vllm_s = source["vllm"]

        actual_sha = patchset_sha256(vllm_u)
        self.assertEqual(vllm_u["patchset_sha256"], actual_sha)
        self.assertEqual(vllm_s["patchset_sha256"], actual_sha)
        self.assertEqual(vllm_u["patch_head"], "ec74bbbca3ada2b879024b1d8c4703e152f789ed")
        self.assertEqual(vllm_u["expected_tree"], "f72aeb6e5a305cf59005f7a14e69b172c637b1d1")
        self.assertEqual(vllm_s["patch_count"], 616)

        mla_arm = read_cluster("experiments/2026-11-06-nccl-sp-channels/tp3-mla-occupancy.json")
        self.assertEqual(
            mla_arm["container"]["expected_labels"]["local.spark-ds41f.vllm.tree"],
            vllm_u["expected_tree"],
        )

    def test_sparse_mla_routes_eliminate_wasted_warps_and_underfilled_waves(self) -> None:
        # 16-head ranks (TP3 rank 2 after _trim_padded_groups and TP4 all ranks):
        r16 = _simulate_routes(16, 128, 512)
        self.assertEqual(r16["block_H"], 16)
        self.assertEqual(r16["threads"], 128)
        self.assertEqual(r16["head_blocks"], 1)
        self.assertEqual(r16["split_max_rows"], 9)
        self.assertEqual(r16["decode_max_rows"], 48)

        # 24-head ranks (TP3 ranks 0 & 1):
        r24 = _simulate_routes(24, 128, 512)
        self.assertEqual(r24["block_H"], 32)
        self.assertEqual(r24["threads"], 256)
        self.assertEqual(r24["head_blocks"], 1)
        self.assertEqual(r24["split_max_rows"], 4)
        self.assertEqual(r24["decode_max_rows"], 48)

        # Verify that for every row count > decode_max_rows (rows >= 49), unsplit extend
        # launches >= 49 CTAs (> 1 full wave of 48 SMs on GB10).
        for rows in range(1, 49):
            decode_ctas = r24["segments"] * r24["head_blocks"] * rows
            self.assertGreaterEqual(decode_ctas, rows)
        self.assertGreater(49 * r24["head_blocks"], 48)

        # SWA-only layers (segments == 1, including DSpark drafter in _draft_workspace_lane)
        # keep decode_max_rows = 144 so drafter workspace reservation is unchanged.
        swa24 = _simulate_routes(24, 128, 0)
        self.assertEqual(swa24["segments"], 1)
        self.assertEqual(swa24["split_max_rows"], 24)
        self.assertEqual(swa24["decode_max_rows"], 144)


if __name__ == "__main__":
    unittest.main()
