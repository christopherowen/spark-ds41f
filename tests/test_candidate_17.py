"""Tests for Candidate 17 (experiments/2026-10-26-spark-ds41f-candidate/rc17):
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
RC17 = ROOT / "experiments/2026-10-26-spark-ds41f-candidate/rc17"
BUNDLE = RC17 / "bundles/candidate-17"
sys.path.insert(0, str(RC17))

import measure_candidate_17 as measure  # noqa: E402


def load_spark():
    loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
    spec = importlib.util.spec_from_loader("spark", loader)
    spark = importlib.util.module_from_spec(spec)
    loader.exec_module(spark)
    return spark


class TestCandidate17(unittest.TestCase):
    def test_patch_series_and_locks_resolve_with_zero_doctor_problems(self) -> None:
        spark = load_spark()
        series_path = RC17 / "vllm/series"
        self.assertTrue(series_path.is_file())
        patches = spark.read_series(series_path)
        # 452 patches in rc14 + 25 new patches (0568..0592) = 478 patches total
        self.assertEqual(len(patches), 478)
        tail_numbers = [Path(p).name[:4] for p in patches[-26:]]
        self.assertEqual(tail_numbers, [f"{n:04d}" for n in range(568, 594)])
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
                f"experiments/2026-10-26-spark-ds41f-candidate/rc17/{cfg_name}"
            )
            nodes = spark.read_json(cluster.get("nodes_config", "config/nodes.json"))
            upstreams = spark.read_json(cluster["upstreams_config"])
            problems = spark.local_doctor(cluster, nodes, upstreams)
            self.assertEqual(problems, [], f"{cfg_name}: {problems}")
            inputs = spark.build_inputs(upstreams)
            self.assertEqual(
                inputs["vllm"]["patch_head"],
                "b8574bf2236383e5855266a7224abe946edbd513",
            )
            self.assertEqual(
                inputs["vllm"]["expected_tree"],
                "1399ec842d35a42038b463f35ea1cab729dd65ca",
            )

    def test_unified_overlays_preserve_all_14_prs(self) -> None:
        adaptive_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
        ).read_text(encoding="utf-8")
        # #686/#687 rework: planned_confidences + slot_draft_limits + block-width non-copy dispatch + mid-prefill confidence preservation
        self.assertIn("planned_confidences: bool = False", adaptive_src)
        self.assertIn("_planned_probs", adaptive_src)
        self.assertIn("slot_draft_limits", adaptive_src)
        self.assertIn("eligible_drafts", adaptive_src)
        self.assertIn("mid_prefill_slots", adaptive_src)
        self.assertNotIn("SPARK_DS41F_DSPARK_PIECEWISE_STEP_COST", adaptive_src)
        self.assertNotIn("SPARK_DS41F_DSPARK_ONLINE_CALIBRATION", adaptive_src)

        runner_src = (
            BUNDLE / "overlay/vllm/v1/worker/gpu/model_runner.py"
        ).read_text(encoding="utf-8")
        # #703/#704 rework: flag-free dead-row estimator clamp + pre-drafter copy prepare_alloc + scatter_rows_calls gather declaration
        self.assertIn("live_rejected", runner_src)
        self.assertIn("self.copy_drafter.prepare_alloc(", runner_src)
        self.assertIn(
            "gather=not self.draft_tokens_handler.track_consumed_drafts,",
            runner_src,
        )

        sched_src = (
            BUNDLE / "overlay/vllm/v1/core/sched/scheduler.py"
        ).read_text(encoding="utf-8")
        # #686/#687 rework: copy drafts lookahead in scheduler gated on non-prefill-chunk steps
        self.assertIn("num_verification_tokens", sched_src)
        self.assertIn("not any(r.is_prefill_chunk for r in self.running)", sched_src)

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
        # #688: local SP row routing + gather_mxfp8
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

        # #711 rework: Vision encoder TP sharding always on across TP ranks (zero feature flags)
        vision_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/nvidia/tilelang_vision.py"
        ).read_text(encoding="utf-8")
        vl_src = (
            BUNDLE / "overlay/vllm/models/deepseek_v4_1/nvidia/vl_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tensor_model_parallel_all_gather", vision_src)
        self.assertIn("use_data_parallel", vl_src)
        self.assertNotIn("SPARK_DS41F_VISION_TP", vision_src)
        self.assertNotIn("SPARK_DS41F_VISION_TP", vl_src)

    def test_verdict_and_results_match(self) -> None:
        verdict = json.loads((RC17 / "verdict.json").read_text(encoding="utf-8"))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["failed"], [])
        self.assertEqual(verdict["errors"], [])

        saved = json.loads((RC17 / "results.json").read_text(encoding="utf-8"))
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
