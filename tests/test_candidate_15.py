"""Tests for Candidate 15 (experiments/2026-10-26-spark-ds41f-candidate/rc15):
unified 20-patch series (0568..0587), source/upstream locks, cluster arms,
overlay integrity across all 14 unified PRs, and recorded verdict/results."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RC15 = ROOT / "experiments/2026-10-26-spark-ds41f-candidate/rc15"
BUNDLE = RC15 / "bundles/candidate-15"
sys.path.insert(0, str(RC15))

import measure_candidate_15 as measure  # noqa: E402


def load_spark():
    loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
    spec = importlib.util.spec_from_loader("spark", loader)
    spark = importlib.util.module_from_spec(spec)
    loader.exec_module(spark)
    return spark


class TestCandidate15(unittest.TestCase):
    def test_patch_series_and_locks_resolve_with_zero_doctor_problems(self) -> None:
        spark = load_spark()
        series_path = RC15 / "vllm/series"
        self.assertTrue(series_path.is_file())
        patches = spark.read_series(series_path)
        # 452 patches in rc14 + 27 new patches (0568..0594) = 479 patches total
        self.assertEqual(len(patches), 479)
        tail_numbers = [Path(p).name[:4] for p in patches[-27:]]
        self.assertEqual(tail_numbers, [f"{n:04d}" for n in range(568, 595)])
        for rel_patch in patches:
            self.assertTrue(
                (series_path.parent / rel_patch).is_file(),
                f"Missing patch in series: {rel_patch}",
            )

        orig_patchset = spark.patchset_sha256
        cache = {}

        def cached_patchset(source: dict) -> str:
            key = source.get("patch_series")
            if key not in cache:
                cache[key] = orig_patchset(source)
            return cache[key]

        spark.patchset_sha256 = cached_patchset
        for cfg_name in ("rc-tp4.json", "rc-tp3.json", "image-dgx4.json"):
            cluster = spark.read_cluster(
                f"experiments/2026-10-26-spark-ds41f-candidate/rc15/{cfg_name}"
            )
            nodes = spark.read_json(cluster.get("nodes_config", "config/nodes.json"))
            upstreams = spark.read_json(cluster["upstreams_config"])
            problems = spark.local_doctor(cluster, nodes, upstreams)
            self.assertEqual(problems, [], f"{cfg_name}: {problems}")
            inputs = spark.build_inputs(upstreams)
            self.assertEqual(
                inputs["vllm"]["patch_head"],
                "02099795a447d164325551bf8e20c3d7b6863a22",
            )
            self.assertEqual(
                inputs["vllm"]["expected_tree"],
                "096391fe4e2928af228c96321a959d1ae560e9e8",
            )

    def test_unified_overlays_preserve_all_14_prs(self) -> None:
        adaptive_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
        ).read_text(encoding="utf-8")
        # #686: seeded per-request budgeting + mid-prefill reset
        self.assertIn("_seeded_sampled", adaptive_src)
        self.assertIn("_req_stale_confidences", adaptive_src)
        # #687: copy drafts planned_confidences
        self.assertIn("planned_confidences: bool = False", adaptive_src)
        self.assertIn("_planned_probs", adaptive_src)
        # #693: piecewise CUDA-graph / 16-row GEMM / 64-row LM-head step cost
        self.assertIn("def padded_capture_sizes(", adaptive_src)
        self.assertIn("def step_costs(", adaptive_src)
        self.assertIn("GRAPH_PAD_COST_FRACTION", adaptive_src)
        # #695: device dead-row capacity clamp
        self.assertIn("_active_dead_row_reqs", adaptive_src)
        self.assertIn("self._dead_row_cut[:num_reqs]", adaptive_src)
        # #700: continuation-value cutoff
        self.assertIn("def continuation_enabled(", adaptive_src)
        self.assertIn("def effective_survival_scores(", adaptive_src)
        self.assertIn("continuation_alpha", adaptive_src)

        spec_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
        ).read_text(encoding="utf-8")
        # #694: online IRLS logistic calibration
        self.assertIn("calibrate_confidence_probs", spec_src)
        # #699: greedy shard runner-up margin
        self.assertIn("greedy_markov_shard_pair", spec_src)
        self.assertIn("margin_weight", spec_src)
        # #701: concurrency-bounded draft depth
        self.assertIn("def active_speculative_steps(", spec_src)
        self.assertIn("SPARK_DS41F_DSPARK_MAX_STEPS_BY_BATCH", spec_src)

        sched_src = (
            BUNDLE / "overlay/vllm/v1/core/sched/scheduler.py"
        ).read_text(encoding="utf-8")
        # #687: copy drafts lookahead in scheduler
        self.assertIn("num_verification_tokens", sched_src)
        # #696: prefill-decode interleave
        self.assertIn("prefill_decode_interleave_steps", sched_src)
        self.assertIn("_interleave_decode_steps_remaining", sched_src)
        # #697: context-aware 2,048-token chunk grid
        self.assertIn("prefill_step_target_s", sched_src)

        loader_src = (
            BUNDLE / "overlay/vllm/model_executor/model_loader/direct_loader.py"
        ).read_text(encoding="utf-8")
        # #698: contiguous weight arena + Order-13 buddy sponge
        self.assertIn("_ARENA_ALIGN_BYTES", loader_src)
        self.assertIn("parse_buddyinfo_order13", loader_src)

        l2_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/l2_prefetch.py"
        ).read_text(encoding="utf-8")
        model_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/nvidia/model.py"
        ).read_text(encoding="utf-8")
        # #705: per-type L2 weight prefetch
        self.assertIn("ENGRAM", l2_src)
        self.assertIn("_l2pf_comp", model_src)
        self.assertIn('("ENGRAM",', model_src)

        sp_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/sp_prefill.py"
        ).read_text(encoding="utf-8")
        # #703: local SP row routing + gather_mxfp8
        self.assertIn("def gather_mxfp8(", sp_src)

        # #706 & #707: indexer q_cast reuse + RoPE-fused MXFP4 quantize + wk(latent)
        attn_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/attention.py"
        ).read_text(encoding="utf-8")
        self.assertIn("indexer.wk(latent)", attn_src)

        # #708: DSpark context main_x cast + kv_norm + rope + cache_writer fusion
        front_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/front.py"
        ).read_text(encoding="utf-8")
        self.assertIn("attention_front", front_src)

        # #710: Fuse mHC post+collapse
        mhc_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/tilelang/mhc.py"
        ).read_text(encoding="utf-8")
        self.assertIn("mhc_post_collapse", mhc_src)

        # Pre-built native io_uring helpers cache
        dio_src = (
            BUNDLE / "overlay/vllm/model_executor/model_loader/direct_io/_native.py"
        ).read_text(encoding="utf-8")
        self.assertIn("VLLM_NATIVE_CACHE_DIR", dio_src)

        # Vision encoder TP sharding (opt-in via SPARK_DS41F_VISION_TP)
        vl_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/nvidia/vl_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("SPARK_DS41F_VISION_TP", vl_src)

    def test_verdict_and_results_match(self) -> None:
        verdict = json.loads((RC15 / "verdict.json").read_text(encoding="utf-8"))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["failed"], [])
        self.assertEqual(verdict["errors"], [])

        saved = json.loads((RC15 / "results.json").read_text(encoding="utf-8"))
        computed = measure.build_summary()
        self.assertEqual(saved, computed)
        self.assertEqual(
            saved["subsystems"]["adaptive_verification"]["seeded_batch_invariance"][
                "match_rate"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
