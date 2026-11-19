"""Tests for DSpark target-to-drafter and intra-draft step-0 top-k selection reuse
(experiments/2026-11-08-dspark-topk-selection-reuse, vLLM patch 0762)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark

EXP_DIR = ROOT / "experiments/2026-11-08-dspark-topk-selection-reuse"
PATCH_NAME = "0762-perf-dsv41-dspark-target-and-anchor-topk-selection-reuse.patch"
EXPECTED_PATCH_HEAD = "c7ae910ad89f089956593ac2ec4d2814c1c5e10f"
EXPECTED_TREE = "4418bf7a9180346291a3b3c38ff53ac718dc1651"
EXPECTED_PATCHSET_SHA256 = (
    "1b3a70ada735472ac8439c99a43b9e258438f75008cce992cd57da0b24686e15"
)

try:
    import torch
except ImportError:  # pragma: no cover - host python without torch
    torch = None  # type: ignore[assignment]


def ref_broadcast_dspark_step0_topk_indices(
    step0_topk: list[list[int]],
    visible_anchor: list[int],
    num_query_per_req: int,
    *,
    num_reqs: int | None = None,
    append_tail: bool = False,
) -> list[list[int]]:
    """Reference implementation of broadcast_dspark_step0_topk_indices."""
    max_reqs = len(step0_topk)
    topk = len(step0_topk[0]) if max_reqs else 0
    if len(visible_anchor) == max_reqs * num_query_per_req and num_query_per_req > 1:
        vis_req = visible_anchor[0::num_query_per_req]
    else:
        vis_req = visible_anchor[:max_reqs]
    limit = max_reqs if num_reqs is None else int(num_reqs)
    out: list[list[int]] = []
    for r in range(max_reqs):
        vis = int(vis_req[r])
        active = (r < limit) and (vis > 0)
        if active:
            valid = [idx for idx in step0_topk[r] if 0 <= idx < vis]
        else:
            valid = []
        valid_count = len(valid)
        compacted = valid + [-1] * (topk - valid_count)
        for s in range(num_query_per_req):
            row = list(compacted)
            if append_tail and num_query_per_req > 1 and active and s > 0:
                base_slot = max(0, min(valid_count, topk - s))
                for t in range(s):
                    if base_slot + t < topk:
                        row[base_slot + t] = vis + t
            out.append(row)
    return out


def ref_gather_dspark_anchor_topk_indices(
    target_topk: list[list[int]],
    target_query_start_loc: list[int],
    num_rejected: list[int] | None,
    visible_anchor: list[int],
    num_reqs: int,
    num_query_per_req: int,
    *,
    target_request_positions: list[int] | None = None,
    draft_request_positions: list[int] | None = None,
    append_tail: bool = False,
) -> list[list[int]]:
    """Reference implementation of gather_dspark_anchor_topk_indices."""
    if (
        num_query_per_req > 1
        and len(visible_anchor) > len(target_query_start_loc) - 1
        and len(visible_anchor) % num_query_per_req == 0
    ):
        vis_req = visible_anchor[0::num_query_per_req]
    else:
        vis_req = list(visible_anchor)
    max_reqs = min(len(vis_req), len(target_query_start_loc) - 1)
    vis_req = vis_req[:max_reqs]
    max_target_row = max(len(target_topk) - 1, 0)
    anchor_rows: list[list[int]] = []
    for r in range(max_reqs):
        active = (r < num_reqs) and (vis_req[r] > 0)
        q_start = int(target_query_start_loc[r])
        q_end = int(target_query_start_loc[r + 1])
        if num_rejected is not None:
            anchor_row = max(q_end - int(num_rejected[r]) - 1, q_start)
        elif target_request_positions is not None and draft_request_positions is not None:
            offset = int(draft_request_positions[r]) - int(target_request_positions[r])
            anchor_row = min(max(q_start + offset, q_start), max(q_end - 1, q_start))
        else:
            anchor_row = max(q_end - 1, q_start)
        safe_row = min(max(anchor_row if active else 0, 0), max_target_row)
        anchor_rows.append(list(target_topk[safe_row]))
    return ref_broadcast_dspark_step0_topk_indices(
        anchor_rows,
        vis_req,
        num_query_per_req,
        num_reqs=num_reqs,
        append_tail=append_tail,
    )


class TestDSparkTopkSelectionReuse(unittest.TestCase):
    def test_patch_series_locks_and_profiles_resolve_cleanly(self) -> None:
        series_path = EXP_DIR / "vllm/series"
        self.assertTrue(series_path.is_file())
        patches = spark.read_series(series_path)
        self.assertEqual(len(patches), 642)
        self.assertEqual(Path(patches[-1]).name, PATCH_NAME)
        for rel_patch in patches:
            self.assertTrue(
                (series_path.parent / rel_patch).is_file(),
                f"Missing patch in series: {rel_patch}",
            )

        orig_patchset = spark.patchset_sha256
        cache: dict[str, str] = {}

        def cached_patchset(source: dict) -> str:
            key = source.get("patch_series")
            if key not in cache:
                cache[key] = orig_patchset(source)
            return cache[key]

        spark.patchset_sha256 = cached_patchset
        try:
            for shape, nodes_path in (
                ("tp4", "config/examples/nodes-ring4.json"),
                ("tp3", "config/nodes.example.json"),
            ):
                cluster = spark.read_cluster(
                    f"experiments/2026-11-08-dspark-topk-selection-reuse/{shape}.json"
                )
                nodes = spark.read_json(nodes_path)
                next(n for n in nodes["nodes"] if n.get("head"))["management_ip"] = (
                    cluster["distributed"]["master_addr"]
                )
                upstreams = spark.read_json(cluster["upstreams_config"])
                problems = spark.local_doctor(cluster, nodes, upstreams)
                self.assertEqual(problems, [], f"{shape}: {problems}")
                inputs = spark.build_inputs(upstreams)
                self.assertEqual(inputs["vllm"]["patch_head"], EXPECTED_PATCH_HEAD)
                self.assertEqual(inputs["vllm"]["expected_tree"], EXPECTED_TREE)
                self.assertEqual(
                    inputs["vllm"]["patchset_sha256"], EXPECTED_PATCHSET_SHA256
                )
        finally:
            spark.patchset_sha256 = orig_patchset

    def test_patch_0762_contains_required_hooks_and_unit_tests(self) -> None:
        patch_text = (EXP_DIR / "vllm" / PATCH_NAME).read_text(encoding="utf-8")
        for required in (
            "def broadcast_dspark_step0_topk_indices(",
            "def gather_dspark_anchor_topk_indices(",
            "def _uses_draft_indexer(",
            "def _dspark_step0_stride(",
            "def _maybe_gather_target_topk(",
            "def prepare_target_topk_reuse(",
            "tests/models/test_deepseek_v4_1_dspark_topk_reuse.py",
            "Upstream base:",
            "Rationale:",
            "Quality implications:",
            "Tests:",
            "Upstream status:",
        ):
            self.assertIn(required, patch_text)

    def test_seq_len_le_topk_and_straddling_boundary_conditions(self) -> None:
        topk = 512
        num_query = 5
        # Request 0: visible_anchor = 128 (well below topk = 512)
        # Request 1: visible_anchor = 510 (straddles topk = 512 across steps 0..4: 510..514)
        row0 = list(range(128)) + [-1] * (topk - 128)
        row1 = list(range(510)) + [-1, -1]
        out_notail = ref_broadcast_dspark_step0_topk_indices(
            [row0, row1], [128, 510], num_query, num_reqs=2, append_tail=False
        )
        out_tail = ref_broadcast_dspark_step0_topk_indices(
            [row0, row1], [128, 510], num_query, num_reqs=2, append_tail=True
        )
        for s in range(num_query):
            self.assertEqual(out_notail[s][:128], list(range(128)))
            self.assertEqual(out_notail[s][128:], [-1] * (topk - 128))
            self.assertEqual(out_tail[s][: 128 + s], list(range(128 + s)))
            self.assertEqual(out_tail[s][128 + s :], [-1] * (topk - 128 - s))

            # Request 1 straddles 512 at s = 2, 3, 4
            expected_len = min(510 + s, topk)
            self.assertEqual(len([x for x in out_tail[num_query + s] if x >= 0]), expected_len)
            if s > 0:
                self.assertEqual(
                    out_tail[num_query + s][expected_len - s : expected_len],
                    list(range(510, 510 + s)),
                )

    def test_seq_len_gt_topk_and_rejected_slots_overwritten_next_step(self) -> None:
        topk = 16
        num_query = 5
        # 3 requests verified with 6 tokens each (query_start_loc = [0, 6, 12, 18])
        target_qsl = [0, 6, 12, 18]
        # Req 0: 0 rejected -> anchor row 5 (visible = 1005)
        # Req 1: 3 rejected -> anchor row 8 (visible = 2002); row 8 has a stale index 2003 at col 0
        # Req 2: 5 rejected (all drafts rejected) -> anchor row 12 (visible = 500)
        num_rejected = [0, 3, 5]
        visible_anchor = [1005, 2002, 500]
        target_req_pos = [1000, 2000, 500]
        draft_req_pos = [1005, 2002, 500]

        target_topk = [[r * 100 + c for c in range(topk)] for r in range(18)]
        # Inject a rejected index 2003 (>= visible_anchor[1] = 2002) at col 0 of anchor row 8
        target_topk[8] = [2003] + [10 + c for c in range(topk - 1)]

        gathered_rej = ref_gather_dspark_anchor_topk_indices(
            target_topk,
            target_qsl,
            num_rejected,
            visible_anchor,
            num_reqs=3,
            num_query_per_req=num_query,
            append_tail=True,
        )
        gathered_pos = ref_gather_dspark_anchor_topk_indices(
            target_topk,
            target_qsl,
            None,
            visible_anchor,
            num_reqs=3,
            num_query_per_req=num_query,
            target_request_positions=target_req_pos,
            draft_request_positions=draft_req_pos,
            append_tail=True,
        )
        self.assertEqual(gathered_rej, gathered_pos)

        # Req 1 step 0: index 2003 (>= 2002) was masked out and compacted left
        self.assertEqual(gathered_rej[num_query + 0][:-1], [10 + c for c in range(topk - 1)])
        self.assertEqual(gathered_rej[num_query + 0][-1], -1)
        # Req 1 step 1 (drafting at pos 2002, where rejected slot 2002 is overwritten):
        # newly drafted position 2002 is appended into the compacted slot, and 2003 remains absent
        self.assertEqual(gathered_rej[num_query + 1][-1], 2002)
        self.assertNotIn(2003, gathered_rej[num_query + 1])

        # Next decode cycle: after rejected slots 2002..2004 are overwritten by new accepted tokens
        # up to visible_anchor = 2004, index 2003 is now < 2004 and becomes valid
        next_step = ref_gather_dspark_anchor_topk_indices(
            target_topk,
            target_qsl,
            [0, 1, 0],
            [1005, 2004, 505],
            num_reqs=3,
            num_query_per_req=num_query,
            append_tail=False,
        )
        # Row 8 tested with visible_anchor = 2004 keeps 2003
        next_step_row8 = ref_broadcast_dspark_step0_topk_indices(
            [target_topk[8]], [2004], num_query, num_reqs=1, append_tail=False
        )
        self.assertIn(2003, next_step_row8[0])
        self.assertEqual(len(next_step), 3 * num_query)

    def test_two_boot_benchmark_receipts_and_quality_gates(self) -> None:
        b1_path = EXP_DIR / "receipts/boot1.json"
        b2_path = EXP_DIR / "receipts/boot2.json"
        p1_path = EXP_DIR / "receipts/pytest-boot1.txt"
        p2_path = EXP_DIR / "receipts/pytest-boot2.txt"
        for p in (b1_path, b2_path, p1_path, p2_path):
            self.assertTrue(p.is_file(), f"Missing receipt: {p}")

        self.assertIn("70 passed", p1_path.read_text(encoding="utf-8"))
        self.assertIn("70 passed", p2_path.read_text(encoding="utf-8"))

        b1 = json.loads(b1_path.read_text(encoding="utf-8"))
        b2 = json.loads(b2_path.read_text(encoding="utf-8"))
        for boot in (b1, b2):
            for shape in ("tp4", "tp3"):
                for conc in ("c1", "c8"):
                    for ctx in ("4096", "16384", "65536", "131072", "262144"):
                        entry = boot["latency"][shape][conc][ctx]
                        self.assertGreater(entry["attn_window_only_ms"], 0.0)
                        if int(ctx) >= 65536:
                            # Target anchor reuse and step-0 anchor reuse must beat full reindexing at >=64K
                            self.assertLess(
                                entry["attn_target_anchor_reuse_ms"],
                                entry["attn_full_reindex_1src_ms"],
                            )
                            self.assertLess(
                                entry["attn_step0_anchor_reuse_ms"],
                                entry["attn_full_reindex_1src_ms"],
                            )
            for ctx in ("4096", "16384", "65536", "131072", "262144"):
                ag = boot["agreement"][ctx]
                self.assertEqual(ag["overlap_pct_by_step_target"][0], 100.0)
                self.assertGreaterEqual(ag["overlap_pct_mean_target"], 97.0)
                self.assertGreaterEqual(ag["cosine_sim_mean_target"], 0.9999)
                self.assertGreaterEqual(ag["top1_agree_pct_mean_target"], 95.0)


if __name__ == "__main__":
    unittest.main()
