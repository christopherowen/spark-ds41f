"""The KV pool's capacity: vLLM's startup figures reproduced from the recipe and the model's
geometry, and what the pool holds once the batches in flight are charged once."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kv_capacity  # noqa: E402

RECIPE = json.loads((ROOT / "config/cluster.json").read_text())
MODEL = json.loads((ROOT / "state/config.json").read_text())["text_config"]


def row(shape: str, **sizing) -> dict:
    cluster = copy.deepcopy(RECIPE)
    cluster["shapes"][shape].setdefault("sizing", {}).update(sizing)
    return kv_capacity.report(cluster, MODEL, [shape])[shape]


class GeometryTest(unittest.TestCase):
    def test_groups_as_vllm_packs_them(self):
        geo = kv_capacity.geometry(MODEL, block_size=256, drafts=5)
        # Layers 2, 8, 14 (ratio 2) and 20 (ratio 1): latents and indexer keys.
        self.assertEqual(geo.pool_block_bytes, 3 * 128 * (288 + 68) + 256 * (288 + 68))
        self.assertEqual(geo.pool_block_bytes, 227_840)
        # 40 target and 3 drafter window caches, three to a block: 20 before the CED
        # decoder (7 groups), 23 from it (8 groups); the boot log's 17 groups.
        self.assertEqual(kv_capacity.ced_decoder_start(MODEL), 20)
        self.assertEqual(geo.window_groups, 15)
        # The drafter's three windows land in three of the decoder's eight groups.
        self.assertEqual(geo.drafter_groups, 3)
        self.assertEqual(geo.ring_groups, 1)


class ReportedTest(unittest.TestCase):
    """The startup line's figures, as the boots logged them."""

    def test_ring(self):
        self.assertEqual(row("tp4-ring", max_num_batched_tokens=8192)["reported_tokens"], 8_580_566)
        self.assertEqual(row("tp4-ring", max_num_batched_tokens=16384)["reported_tokens"], 6_512_700)

    def test_triangle(self):
        self.assertEqual(row("tp3-triangle")["reported_tokens"], 4_765_898)
        self.assertEqual(row("tp3-triangle-4k")["reported_tokens"], 1_348_708)

    def test_r6v_half_million_context(self):
        # r6v's TP3 recipe: 3.5 GiB at 524,288 tokens.
        reported = row("tp3-triangle", max_model_len=524_288, kv_cache_memory_bytes=3_758_096_384)
        self.assertEqual(reported["reported_tokens"], 2_845_543)


class CapacityTest(unittest.TestCase):
    def test_ring_holds_eleven_at_either_budget(self):
        for budget, transient, each in ((8192, 1890, 749_568), (16384, 3810, 718_848)):
            with self.subTest(budget=budget):
                ring = row("tp4-ring", max_num_batched_tokens=budget)
                self.assertEqual(ring["pool_blocks"], 49_483)
                # 4,096 compressed blocks, three per window group, one ring block.
                self.assertEqual(ring["resident_blocks"], 4096 + 15 * 3 + 1)
                self.assertEqual(ring["prompt_blocks_in_flight"], transient)
                self.assertEqual(ring["full_contexts"], 11)
                self.assertGreater(ring["contexts"], ring["reported_contexts"] + 2.8)
                self.assertEqual(ring["every_sequence_tokens"], each)

    def test_triangle(self):
        triangle = row("tp3-triangle")
        self.assertEqual(triangle["full_contexts"], 5)
        self.assertEqual(triangle["every_sequence_tokens"], 697_600)
        small = row("tp3-triangle-4k")
        self.assertEqual(small["full_contexts"], 8)
        self.assertEqual(small["every_sequence_tokens"], 262_144)

    def test_vllm_kv_capacity_figure(self):
        # vLLM 0491 after 0492: the window per request, both batches once, and the
        # drafter's groups one more token (the next candidate's startup line, computed
        # with vLLM's grouping and capacity code on 0492 + 0500-0503 + 0491).
        self.assertAlmostEqual(row("tp4-ring", max_num_batched_tokens=8192)["vllm_capacity_contexts"], 11.48, 2)
        self.assertAlmostEqual(row("tp4-ring", max_num_batched_tokens=16384)["vllm_capacity_contexts"], 11.02, 2)
        self.assertEqual(row("tp4-ring", max_num_batched_tokens=16384)["vllm_capacity_tokens"], 11_554_122)
        self.assertEqual(row("tp3-triangle")["vllm_capacity_tokens"], 5_609_861)

    def test_budget_override(self):
        rows = kv_capacity.report(RECIPE, MODEL, ["tp4-ring"], budget=16384)
        self.assertEqual(rows["tp4-ring"]["reported_tokens"], 6_512_700)


if __name__ == "__main__":
    unittest.main()
