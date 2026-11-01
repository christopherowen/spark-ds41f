"""An experiment's upstream lock, source manifest, arm configs and results receipts pin
the same revisions, patch counts, patch heads, trees and patch ranges."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_source_locks_test", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)

HISTORICAL_PROFILES = {
    "experiments/2026-10-03-tilelang-tp4-performance/round2-tilelang.json",
    "experiments/2026-10-03-tilelang-tp4-performance/tilelang.json",
    "experiments/2026-10-04-tilelang-1m/diagnostics/ctx0.json",
    "experiments/2026-10-04-tilelang-1m/diagnostics/seqs12.json",
    "experiments/2026-10-04-tilelang-1m/diagnostics/seqs16-4k.json",
    "experiments/2026-10-04-tilelang-1m/diagnostics/shapes8.json",
    "experiments/2026-10-23-next-release/rc2/cluster.json",
    "experiments/2026-10-23-next-release/rc3/cluster.json",
}

# Early TileLang experiments whose locks pointed at the mutable root series
# (patches/vllm/series, patches/b12x/series) before per-experiment series existed.
HISTORICAL_ROLLING_SERIES_LOCKS = {
    "experiments/2026-10-03-packed-bf16-head/tilelang-tp4/upstreams.lock.json",
    "experiments/2026-10-04-tilelang-gate-router/upstreams.lock.json",
    "experiments/2026-10-04-tilelang-mhc/upstreams.lock.json",
    "experiments/2026-10-04-tilelang-router/upstreams.lock.json",
    "experiments/2026-10-04-tilelang-sparknet/upstreams.lock.json",
    "experiments/2026-10-04-tilelang-vocab-heads/upstreams.lock.json",
}


class SourceLockTest(unittest.TestCase):
    def test_locks_and_manifests_pin_the_same_revisions_and_identifiers(self) -> None:
        checked, problems = 0, []
        lock_paths = [ROOT / "upstreams.lock.json", *sorted((ROOT / "experiments").rglob("upstreams.lock.json"))]
        for lock_path in lock_paths:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            manifest_path = ROOT / lock.get("source_manifest", "")
            if not lock.get("source_manifest") or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checked += 1
            rel = lock_path.relative_to(ROOT).as_posix()
            for problem in spark.source_identifier_problems(lock, manifest):
                if rel in HISTORICAL_ROLLING_SERIES_LOCKS and "patch-set fingerprint differs" in problem:
                    continue
                problems.append(f"{rel}: {problem}")
        self.assertGreater(checked, 0)
        self.assertEqual(problems, [])

    def test_arm_notes_and_expected_labels_match_locks_and_manifests(self) -> None:
        checked, problems = 0, []
        for path in sorted((ROOT / "experiments").rglob("*.json")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in HISTORICAL_PROFILES or "/results/" in rel or "/bundles/" in rel or "/release-tests/" in rel:
                continue
            try:
                if path.stat().st_size > 65536:
                    continue
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            if not spark.cluster_recipe.is_overlay(raw) and (
                "container" not in raw or "serve_args" not in raw
            ):
                continue
            try:
                cluster = spark.read_cluster(rel) if spark.cluster_recipe.is_overlay(raw) else raw
            except SystemExit:
                continue
            upstreams_rel = cluster.get("upstreams_config", "upstreams.lock.json")
            upstreams_path = ROOT / upstreams_rel
            if not upstreams_path.is_file():
                continue
            lock = json.loads(upstreams_path.read_text(encoding="utf-8"))
            manifest_path = ROOT / lock.get("source_manifest", "")
            if not lock.get("source_manifest") or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checked += 1
            for problem in spark.arm_note_problems(cluster, lock, manifest):
                problems.append(f"{rel}: {problem}")
            if cluster.get("kernel_backend") == "tilelang":
                for problem in spark.kernel_backend.source_problems(cluster, lock, manifest):
                    problems.append(f"{rel}: {problem}")
        self.assertGreater(checked, 0)
        self.assertEqual(problems, [])

    def test_candidate_results_and_verdicts_align_with_source_manifest(self) -> None:
        checked, problems = 0, []
        for results_path in sorted((ROOT / "experiments").rglob("candidate-*/results.json")):
            cand_dir = results_path.parent
            source_path = cand_dir / "source.json"
            if not source_path.is_file():
                continue
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(results, dict) or "patch_head" not in results:
                continue
            source = json.loads(source_path.read_text(encoding="utf-8")).get("vllm", {})
            rel = results_path.relative_to(ROOT).as_posix()
            checked += 1
            for key in ("patch_head", "expected_tree", "patchset_sha256"):
                if key in results and key in source and results[key] != source[key]:
                    problems.append(f"{rel}: {key} {results[key]} != source.json {source[key]}")
            if "total_patches" in results and results["total_patches"] != source.get("patch_count"):
                problems.append(
                    f"{rel}: total_patches {results['total_patches']} != source.json patch_count {source.get('patch_count')}"
                )
            verdict_path = cand_dir / "verdict.json"
            if verdict_path.is_file():
                verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
                if "image_digest" in results and verdict.get("image") and results["image_digest"] != verdict["image"]:
                    problems.append(
                        f"{rel}: image_digest {results['image_digest']} != verdict.json image {verdict['image']}"
                    )
        self.assertGreater(checked, 0)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
