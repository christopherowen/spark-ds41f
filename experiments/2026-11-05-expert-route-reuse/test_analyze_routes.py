import copy
import unittest
from pathlib import Path
import tempfile

from analyze_routes import analyze, projection, summarize, tile_height


def trace(ids, requests=None, kinds=None):
    return {
        "workload": "synthetic-contract", "evidence_kind": "synthetic",
        "capture": "boot-a", "step": 1, "layer": "target.2",
        "source_tree": "test-only", "model_revision": "test-only",
        "experts": 384, "topk": 6,
        "rows": [{"ids": row, "request": requests[i] if requests else "r0",
                  "kind": kinds[i] if kinds else "draft"} for i, row in enumerate(ids)],
    }


class AccountingTest(unittest.TestCase):
    def test_grouping_already_reuses_all_six_speculative_rows(self):
        record = trace([list(range(6))] * 6)
        before = copy.deepcopy(record)
        result = summarize(record, 4, "candidate-38")
        self.assertEqual((result["distinct_experts"], result["used_tiles"]), (6, 6))
        self.assertEqual(result["padding_rows_in_used_tiles"], 60)
        p = result["payload"]["all_2bit"]
        self.assertEqual(p["repeated_expert_payload_bytes"], 0)
        self.assertEqual(p["already_saved_by_grouping_bytes"], 5 * p["grouped_load_payload_bytes"])
        self.assertEqual(record, before, "analysis must not reorder or reroute inputs")

    def test_distinct_requests_can_reuse_within_each_request_only(self):
        ids = [list(range(req * 6, req * 6 + 6)) for req in range(8) for _ in range(6)]
        requests = [f"r{req}" for req in range(8) for _ in range(6)]
        r = summarize(trace(ids, requests), 3, "candidate-38")
        self.assertEqual((r["distinct_experts"], r["used_tiles"]), (48, 48))
        self.assertEqual(r["cross_request_duplicate_experts"], 0)
        self.assertTrue(all(p["jaccard"] == 0 for p in r["request_pair_overlap"]))

    def test_low_overlap_has_no_weight_reuse_despite_large_padding(self):
        ids = [list(range(row * 6, row * 6 + 6)) for row in range(48)]
        r = summarize(trace(ids), 4, "candidate-38")
        self.assertEqual(r["distinct_experts"], 288)
        self.assertEqual(r["used_tiles"], 288)
        self.assertEqual(r["active_row_fraction"], 1 / 16)
        self.assertEqual(r["payload"]["all_4bit"]["already_saved_by_grouping_bytes"], 0)

    def test_hot_expert_crosses_tile_boundary_without_changing_union(self):
        r = summarize(trace([list(range(6))] * 17), 3, "candidate-38")
        self.assertEqual(r["distinct_experts"], 6)
        self.assertEqual(r["used_tiles"], 12)
        p = r["payload"]["all_2bit"]
        self.assertEqual(p["repeated_expert_payload_bytes"], p["unique_expert_payload_bytes"])
        self.assertEqual(r["other_tile_heights"]["32"]["used_tiles"], 6)

    def test_graph_padding_selects_tier_but_does_not_load_experts(self):
        ids = [list(range(6))] + [[-1] * 6] * 1023
        kinds = ["anchor"] + ["padding"] * 1023
        r = summarize(trace(ids, kinds=kinds), 4, "candidate-38")
        self.assertEqual((r["tile_height"], r["live_rows"], r["used_tiles"]), (32, 1, 6))
        self.assertGreater(r["empty_tiles"], 0)
        dead = summarize(trace([[-1] * 6], kinds=["dead"]), 3, "candidate-38")
        self.assertEqual(dead["payload"]["all_2bit"]["grouped_load_payload_bytes"], 0)

    def test_tp4_weight_tail_is_exact_but_scale_code_word_has_unused_bits(self):
        down = projection(5120, 576, 2)
        self.assertEqual(down["fp4_weight"], 1474560)
        self.assertEqual(down["scale_codes"], 25600)
        self.assertEqual(down["unused_scale_code_bits_as_bytes"], 2560)
        self.assertEqual(projection(5120, 768, 2)["unused_scale_code_bits_as_bytes"], 0)

    def test_per_expert_widths_give_exact_payload_instead_of_bounds(self):
        r = trace([list(range(6))])
        r["scale_bits"] = {"4": {"gate_up": [2] * 384, "down": [4] * 384}}
        out = summarize(r, 4, "candidate-38")["payload"]
        expected = 6 * (projection(1152, 5120, 2)["total"] + projection(5120, 576, 4)["total"])
        self.assertEqual(list(out), ["exact"])
        self.assertEqual(out["exact"]["grouped_load_payload_bytes"], expected)

    def test_invalid_or_unmasked_routes_are_refused(self):
        for ids, kind in [([0, 0, 1, 2, 3, 4], "draft"), ([0, 1, 2, 3, 4, 384], "draft"),
                          ([0, 1, 2, 3, 4, 5], "dead")]:
            with self.subTest(ids=ids, kind=kind), self.assertRaises(ValueError):
                summarize(trace([ids], kinds=[kind]), 4, "candidate-38")

    def test_unsupported_geometry_and_empty_captures_are_refused(self):
        for tp, policy in ((2, "candidate-38"), (4, "unknown")):
            with self.assertRaises(ValueError):
                summarize(trace([list(range(6))]), tp, policy)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("")
            with self.assertRaises(ValueError):
                analyze(path)

    def test_tier_boundaries_match_target_and_drafter(self):
        self.assertEqual([tile_height(r, 384, 6, "candidate-38") for r in (1023, 1024, 2047, 2048)], [16, 32, 32, 64])
        self.assertEqual([tile_height(r, 128, 3, "candidate-38") for r in (682, 683, 1365, 1366)], [16, 32, 32, 64])
        self.assertEqual(tile_height(96, 384, 6, "1030"), 32)


if __name__ == "__main__":
    unittest.main()
