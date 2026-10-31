"""Tests for Candidate 18 (experiments/2026-10-26-spark-ds41f-candidate/rc18):
unified 499-patch vLLM series (0001..0614) + sparknet 0001, source/upstream locks,
cluster arms, overlay integrity across all 35 unified PRs, and recorded verdict/results."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RC18 = ROOT / "experiments/2026-10-26-spark-ds41f-candidate/rc18"
BUNDLE = RC18 / "bundles/candidate-18"
sys.path.insert(0, str(RC18))

import measure_candidate_18 as measure  # noqa: E402


def load_spark():
    loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
    spec = importlib.util.spec_from_loader("spark", loader)
    spark = importlib.util.module_from_spec(spec)
    loader.exec_module(spark)
    return spark


class TestCandidate18(unittest.TestCase):
    def test_patch_series_and_locks_resolve_with_zero_doctor_problems(self) -> None:
        spark = load_spark()
        series_path = RC18 / "vllm/series"
        self.assertTrue(series_path.is_file())
        patches = spark.read_series(series_path)
        # 478 patches in rc17 + 21 new patches (0594..0614) = 499 patches total
        self.assertEqual(len(patches), 499)
        tail_numbers = [Path(p).name[:4] for p in patches[-21:]]
        self.assertEqual(tail_numbers, [f"{n:04d}" for n in range(594, 615)])
        for rel_patch in patches:
            self.assertTrue(
                (series_path.parent / rel_patch).is_file(),
                f"Missing patch in series: {rel_patch}",
            )

        sparknet_series = RC18 / "sparknet/series"
        self.assertTrue(sparknet_series.is_file())
        spatches = spark.read_series(sparknet_series)
        self.assertEqual(len(spatches), 1)

        orig_patchset = spark.patchset_sha256
        cache = {}

        def cached_patchset(source: dict) -> str:
            key = source.get("patch_series")
            if key not in cache:
                cache[key] = orig_patchset(source)
            return cache[key]

        spark.patchset_sha256 = cached_patchset
        for cfg_name in ("rc-tp4.json", "rc-tp3.json", "rc-tp3-copy.json", "image-dgx4.json"):
            cluster = spark.read_cluster(
                f"experiments/2026-10-26-spark-ds41f-candidate/rc18/{cfg_name}"
            )
            nodes = spark.read_json(cluster.get("nodes_config", "config/nodes.json"))
            upstreams = spark.read_json(cluster["upstreams_config"])
            problems = spark.local_doctor(cluster, nodes, upstreams)
            self.assertEqual(problems, [], f"{cfg_name}: {problems}")
            inputs = spark.build_inputs(upstreams)
            self.assertEqual(
                inputs["vllm"]["patch_head"],
                "68fe7a9070422c4c5fe68bb820f6a8defd72e408",
            )
            self.assertEqual(
                inputs["vllm"]["expected_tree"],
                "db68f75be55ca74a495a31fde706b07317013d6d",
            )
            self.assertEqual(
                inputs["sparknet"]["patch_head"],
                "fad37ed4802638bad0cd0cb572f5aa1d6fef27bc",
            )
            self.assertEqual(
                inputs["sparknet"]["expected_tree"],
                "104504a654552fa3eefcef1377e871591917f0ea",
            )

    def test_unified_overlays_preserve_all_rc18_prs_without_feature_flags(self) -> None:
        # #687/#619 Rework #2: non-blocking sync_slot_draft_limits + -1 placeholder drafts + pending-capped live drafts
        copy_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/spec_decode/copy_drafts.py"
        ).read_text(encoding="utf-8")
        tcopy_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self._limits_event.query()", copy_src)
        self.assertNotIn("self._limits_event.synchronize()", copy_src)
        self.assertIn("_select_alloc", copy_src)
        self.assertIn("T.int64(-1)", tcopy_src)
        self.assertIn("T.min(pending, T.min(verified[r], cut[r]))", tcopy_src)

        # #717: inline single-rank headless worker without VLLM_HEADLESS_SPAWN_WORKER flag
        serve_src = (
            BUNDLE / "overlay/vllm/v1/executor/multiproc_executor.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_use_inline_headless_worker", serve_src)
        self.assertNotIn("VLLM_HEADLESS_SPAWN_WORKER", serve_src)

        # #727: compact 2-bit and 4-bit expert scale codes in one buffer
        moe_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/moe.py"
        ).read_text(encoding="utf-8")
        self.assertIn("pack_expert_scale_codes", moe_src)

        # #729: 128-row packed vocab head tile and fused greedy epilogue
        vocab_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/vocab.py"
        ).read_text(encoding="utf-8")
        self.assertIn("project_head_greedy_pairs", vocab_src)

        # #730: single-launch project_and_finalize_streams and barrier-free finalize
        mhc_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/mhc.py"
        ).read_text(encoding="utf-8")
        self.assertIn("project_and_finalize_streams", mhc_src)

        # #731: 128x128 tile-contiguous FP8 decode weights
        linear_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/linear.py"
        ).read_text(encoding="utf-8")
        self.assertIn("pack_weight_tiles", linear_src)

        # #732 & #713: attention aux_stream overlap and activation lifetime trimming
        attn_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/attention.py"
        ).read_text(encoding="utf-8")
        self.assertIn("maybe_execute_in_parallel", attn_src)

        # #733: TP-invariant local confidence
        dspark_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tp_invariant_confidence", dspark_src)

        # #735: sparknet.oneshot SP reduce_scatter, exchange, all_to_all_rows
        coll_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/collectives.py"
        ).read_text(encoding="utf-8")
        self.assertIn("roce_comm", coll_src)

        # #736: custom_op wrappers removed from kernels.py
        kernels_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/kernels.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("direct_register_custom_op", kernels_src)

    def test_verdict_and_results_match(self) -> None:
        verdict = json.loads((RC18 / "verdict.json").read_text(encoding="utf-8"))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["failed"], [])
        self.assertEqual(verdict["errors"], [])

        saved = json.loads((RC18 / "results.json").read_text(encoding="utf-8"))
        computed = measure.build_summary()
        self.assertEqual(saved, computed)


if __name__ == "__main__":
    unittest.main()
