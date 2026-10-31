#!/usr/bin/env python3
"""Render manifest-driven status tables and the interactive GitHub Pages dashboard.

Usage:
    scripts/render_docs.py [--check] [--site-dir DIR]

Reads the authoritative machine-readable repository state:
- ``config/cluster.json`` (resolved via ``scripts/recipe.py`` and ``scripts/kv_capacity.py``)
- ``upstreams.lock.json``
- ``state/config.json``
- ``manifests/baselines/*.json``
- ``manifests/sources/*.json``
- ``manifests/suites/*/``

and renders:
1. Deterministic markdown status & benchmark tables between
   ``<!-- BEGIN GENERATED: <id> -->`` and ``<!-- END GENERATED: <id> -->`` markers in
   ``README.md``, ``docs/current-state.md``, and ``TODO.md``.
2. The self-contained interactive GitHub Pages dashboard at ``docs/index.html``
   (plus ``docs/.nojekyll``, and ``<site-dir>/index.html`` + ``<site-dir>/data.json``
   when ``--site-dir`` is given).
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import kv_capacity  # noqa: E402
import recipe  # noqa: E402
from servebench import stats as sb_stats  # noqa: E402

MARKER_RE = re.compile(
    r"(<!-- BEGIN GENERATED: ([a-z0-9_-]+) -->)\n.*?(<!-- END GENERATED: \2 -->)",
    re.DOTALL,
)


def _short_release_name(baseline_stem: str) -> str:
    """Extract the short release identifier from a baseline or suite stem."""
    m = re.match(r"^\d{4}-\d{2}-\d{2}-(?:karmic-kraken-)?(.+)$", baseline_stem)
    return m.group(1) if m else baseline_stem


def _baseline_date(baseline_stem: str) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-", baseline_stem)
    return m.group(1) if m else ""


def _format_gib(num_bytes: int) -> str:
    gib = num_bytes / (1024**3)
    if abs(gib - round(gib, 1)) < 1e-4:
        return f"{gib:.1f} GiB"
    if abs(gib - round(gib, 2)) < 1e-4:
        return f"{gib:.2f} GiB"
    return f"{gib:.5g} GiB"


def _count_series_patches(series_rel_path: str | None) -> int | None:
    if not series_rel_path:
        return None
    path = ROOT / series_rel_path
    if not path.is_file():
        return None
    return sum(
        1
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _parse_run_step(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    out_tps = summary.get("output_tps", {})
    server = summary.get("server", {})
    latency = summary.get("latency", {})
    ttft = latency.get("ttft_s", {})
    tpot = latency.get("tpot_ms", {})
    user_tps = latency.get("per_user_tps", {})
    slo = latency.get("slo", {})
    return {
        "concurrency": data.get("workload", {}).get("concurrency"),
        "mode": data.get("workload", {}).get("mode", "closed"),
        "rate": data.get("workload", {}).get("rate"),
        "output_tps": round(out_tps.get("mean", 0.0), 2) if out_tps else None,
        "output_tps_ci95_pct": round(out_tps.get("ci95_pct", 0.0), 2) if out_tps.get("ci95_pct") is not None else None,
        "steps_per_s": round(server.get("steps_per_s", {}).get("mean", 0.0), 2) if server.get("steps_per_s") else None,
        "output_tokens_per_step": round(server.get("output_tokens_per_step", {}).get("mean", 0.0), 3)
        if server.get("output_tokens_per_step")
        else None,
        "accepted_per_draft": round(server.get("accepted_per_draft", {}).get("mean", 0.0), 3)
        if server.get("accepted_per_draft")
        else None,
        "scheduled_tokens_per_step": round(server.get("scheduled_tokens_per_step", {}).get("mean", 0.0), 2)
        if server.get("scheduled_tokens_per_step")
        else None,
        "ttft_p50_s": round(ttft.get("p50", 0.0), 3) if ttft.get("p50") is not None else None,
        "ttft_p99_s": round(ttft.get("p99", 0.0), 3) if ttft.get("p99") is not None else None,
        "tpot_p50_ms": round(tpot.get("p50", 0.0), 2) if tpot.get("p50") is not None else None,
        "tpot_p99_ms": round(tpot.get("p99", 0.0), 2) if tpot.get("p99") is not None else None,
        "per_user_tps_p50": round(user_tps.get("p50", 0.0), 2) if user_tps.get("p50") is not None else None,
        "goodput_rps": round(slo.get("goodput_rps", 0.0), 3) if slo.get("goodput_rps") is not None else None,
        "slo_attainment_pct": round(slo.get("attainment_pct", 0.0), 1) if slo.get("attainment_pct") is not None else None,
        "survival": [
            round(server.get(f"survival_at_{k}", {}).get("mean", 0.0), 3)
            for k in range(5)
            if server.get(f"survival_at_{k}")
        ],
        "coverage": [
            round(server.get(f"coverage_at_{k}", {}).get("mean", 0.0), 3)
            for k in range(5)
            if server.get(f"coverage_at_{k}")
        ],
    }


def _parse_grid_step(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    shapes = []
    decode_means = []
    for row in data.get("engine", []):
        samples = row.get("samples") or []
        if samples:
            kept, _ = sb_stats.trim_slow(samples)
            trimmed_mean = sum(kept) / len(kept)
        else:
            trimmed_mean = row.get("step_ms", {}).get("mean", 0.0)
        is_prefill = bool(row.get("shape", {}).get("prefill"))
        if not is_prefill and trimmed_mean > 0:
            decode_means.append(trimmed_mean)
        shapes.append(
            {
                "label": row["label"],
                "num_reqs": row.get("shape", {}).get("num_reqs"),
                "num_tokens": row.get("shape", {}).get("num_tokens"),
                "context_len": row.get("shape", {}).get("context_len"),
                "step_ms": round(trimmed_mean, 3),
                "ci95_pct": round(row.get("step_ms", {}).get("ci95_pct", 0.0), 2)
                if row.get("step_ms", {}).get("ci95_pct") is not None
                else None,
            }
        )
    geomean = math.exp(sum(math.log(m) for m in decode_means) / len(decode_means)) if decode_means else None
    return {
        "shape_count": len(decode_means),
        "geomean_ms": round(geomean, 3) if geomean is not None else None,
        "shapes": shapes,
    }


def _parse_prefill_step(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    out = []
    for row in data.get("prefill", []):
        tps = row.get("tps", {})
        out.append(
            {
                "size": row["size"],
                "prompt_tokens": row.get("prompt_tokens"),
                "tps": round(tps.get("mean", 0.0), 1),
                "ci95_pct": round(tps.get("ci95_pct", 0.0), 2) if tps.get("ci95_pct") is not None else None,
            }
        )
    return out


def _parse_stall_step(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    rows = data.get("stall", [])
    if not rows:
        return None
    first = rows[0]
    return {
        "streams": data.get("workload", {}).get("streams"),
        "size": first.get("size"),
        "prefill_tps": round(first.get("tps", {}).get("mean", 0.0), 1) if first.get("tps") else None,
        "during_mean_s": round(first.get("during_mean_s", {}).get("mean", 0.0), 3)
        if first.get("during_mean_s")
        else None,
        "during_p90_s": round(first.get("during_p90_s", {}).get("mean", 0.0), 3)
        if first.get("during_p90_s")
        else None,
        "during_p99_s": round(first.get("during_p99_s", {}).get("mean", 0.0), 3)
        if first.get("during_p99_s")
        else None,
        "probe_ttft_p50_s": round(first.get("probe_ttft_p50_s", {}).get("mean", 0.0), 3)
        if first.get("probe_ttft_p50_s")
        else None,
        "probe_ttft_max_s": round(first.get("probe_ttft_max_s", {}).get("mean", 0.0), 3)
        if first.get("probe_ttft_max_s")
        else None,
    }


def _parse_verdicts_text(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    verdicts = []
    in_verdicts = False
    for line in lines:
        if line.startswith("verdicts:"):
            in_verdicts = True
            continue
        if in_verdicts:
            if not line.strip() or line.startswith("paired rows:") or line.startswith("########"):
                break
            if line.lstrip().startswith("metric"):
                continue
            m = re.match(
                r"^(.+?)\s+([+-]\d+\.\d+%)\s+(\d+\.\d+%)\s+([+-]\d+\.\d+%)\s+(?:(\S+)\s+(\d+\.\d+%)\s+)?(better|same|worse.*)$",
                line.strip(),
            )
            if m:
                verdicts.append(
                    {
                        "metric": m.group(1).strip(),
                        "change": m.group(2),
                        "ci95": m.group(3),
                        "repeat": m.group(4),
                        "verdict": m.group(7).strip(),
                    }
                )
    return verdicts


def load_suite_directory(suite_dir: Path) -> dict[str, Any]:
    suite_meta = json.loads((suite_dir / "suite.json").read_text())
    stem = suite_dir.name
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-(?:karmic-kraken-)?(.+)-(tp[34])$", stem)
    date_str = m.group(1) if m else ""
    release_name = m.group(2) if m else stem
    topology = m.group(3) if m else ("tp4" if "tp4" in stem else "tp3")

    runs: dict[str, Any] = {}
    for step_id in ("c16", "c8", "c4", "c1", "poisson"):
        parsed = _parse_run_step(suite_dir / f"{step_id}.json")
        if parsed:
            runs[step_id] = parsed

    single_dir = suite_dir.parent / f"{stem}-single-stream"
    if "c1" not in runs and (single_dir / "c1.json").is_file():
        runs["c1"] = _parse_run_step(single_dir / "c1.json")

    sampled: dict[str, Any] = {}
    sampled_dir = suite_dir.parent / f"{stem}-sampled"
    if sampled_dir.is_dir():
        for step_id in ("c16", "c8", "c1"):
            parsed = _parse_run_step(sampled_dir / f"{step_id}.json")
            if parsed:
                sampled[step_id] = parsed

    stalls: dict[str, Any] = {}
    for stall_id in ("stall-3", "stall-6", "stall-7", "stall-11", "stall-15"):
        parsed_stall = _parse_stall_step(suite_dir / f"{stall_id}.json")
        if parsed_stall:
            stalls[stall_id] = parsed_stall

    verdicts = _parse_verdicts_text(suite_dir / "release-table.txt")
    if (single_dir / "release-table.txt").is_file():
        verdicts.extend(_parse_verdicts_text(single_dir / "release-table.txt"))

    return {
        "id": stem,
        "path": f"manifests/suites/{stem}",
        "date": date_str,
        "release": release_name,
        "topology": topology.upper(),
        "suite_version": suite_meta.get("suite", {}).get("version", 1),
        "started_utc": suite_meta.get("started_utc"),
        "image": suite_meta.get("metadata", {}).get("identity", {}).get("image"),
        "transport": suite_meta.get("metadata", {}).get("identity", {}).get("transport"),
        "runs": runs,
        "sampled": sampled,
        "grid": _parse_grid_step(suite_dir / "grid.json"),
        "prefill": _parse_prefill_step(suite_dir / "prefill.json"),
        "stalls": stalls,
        "verdicts": verdicts,
    }


def collect_manifest_data() -> dict[str, Any]:
    cluster = json.loads((ROOT / "config/cluster.json").read_text())
    upstreams = json.loads((ROOT / "upstreams.lock.json").read_text())
    model_cfg = json.loads((ROOT / "state/config.json").read_text())
    kv_rows = kv_capacity.report(cluster, model_cfg.get("text_config", model_cfg), [])

    baselines_by_stem: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "manifests/baselines").glob("*.json")):
        baselines_by_stem[path.stem] = json.loads(path.read_text())

    sources_by_stem: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "manifests/sources").glob("*.json")):
        sources_by_stem[path.stem] = json.loads(path.read_text())

    suites: list[dict[str, Any]] = []
    suites_by_path: dict[str, dict[str, Any]] = {}
    for suite_dir in sorted((ROOT / "manifests/suites").iterdir()):
        if not suite_dir.is_dir() or suite_dir.name.endswith(("-sampled", "-single-stream")):
            continue
        parsed = load_suite_directory(suite_dir)
        suites.append(parsed)
        suites_by_path[parsed["path"]] = parsed

    promoted_stem = cluster["promoted_baseline"]
    promoted_manifest = baselines_by_stem[promoted_stem]
    promoted_source = sources_by_stem.get(f"{promoted_stem}-source", {})

    shapes_info: list[dict[str, Any]] = []
    for shape_name in cluster["shapes"]:
        entry = recipe.shape_entry(cluster, shape_name)
        resolved = recipe.resolve(cluster, shape_name)
        vals = recipe.values(cluster, entry)
        kv = kv_rows[shape_name]
        bench_ref = entry.get("benchmark_reference")
        ref_suite = suites_by_path.get(bench_ref) if bench_ref else None
        qualified_release = (
            ref_suite["release"]
            if ref_suite
            else (_short_release_name(Path(bench_ref).stem.rsplit("-", 1)[0]) if bench_ref else None)
        )
        shapes_info.append(
            {
                "name": shape_name,
                "default": shape_name == cluster.get("default_shape"),
                "nodes": int(entry["nodes"]),
                "transport": entry["fabric"]["transport"],
                "kernel_release": resolved["host"]["kernel"]["release"],
                "page_size_bytes": int(resolved["host"]["kernel"]["page_size"]),
                "page_size_label": f"{int(resolved['host']['kernel']['page_size']) // 1024} KiB",
                "memory_saver_required": bool(resolved["host"]["kernel"]["memory_saver_required"]),
                "display_carveout": resolved["environment"].get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS") == "1",
                "max_model_len": int(entry["sizing"]["max_model_len"]),
                "max_num_seqs": int(entry["sizing"]["max_num_seqs"]),
                "max_num_batched_tokens": int(entry["sizing"]["max_num_batched_tokens"]),
                "long_prefill_token_threshold": int(vals["long_prefill_token_threshold"]),
                "max_cudagraph_capture_size": int(vals["max_cudagraph_capture_size"]),
                "kv_cache_memory_bytes": int(entry["sizing"]["kv_cache_memory_bytes"]),
                "kv_cache_gib": _format_gib(int(entry["sizing"]["kv_cache_memory_bytes"])),
                "prefix_cache_retention_interval": int(entry["sizing"]["prefix_cache_retention_interval"]),
                "kv_capacity": kv,
                "benchmark_reference": bench_ref,
                "qualified_release": qualified_release,
            }
        )

    upstreams_summary: list[dict[str, Any]] = []
    src_id = promoted_manifest.get("source_identity", {})
    for key, item in upstreams.get("sources", {}).items():
        series = item.get("patch_series")
        patch_count = _count_series_patches(series)
        tree_key = f"{key}_tree"
        patch_head_key = f"{key}_patch_head"
        upstreams_summary.append(
            {
                "name": key,
                "kind": item.get("kind"),
                "upstream": item.get("upstream"),
                "version": item.get("version") or item.get("release_tag") or item.get("tag"),
                "revision": item.get("revision") or item.get("source_revision"),
                "patch_series": series,
                "patch_count": patch_count,
                "patch_head": src_id.get(patch_head_key),
                "tree": src_id.get(tree_key),
                "role": item.get("role"),
            }
        )

    timeline: list[dict[str, Any]] = []
    for stem, b in reversed(list(baselines_by_stem.items())):
        timeline.append(
            {
                "stem": stem,
                "short_name": _short_release_name(stem),
                "date": _baseline_date(stem),
                "captured_at": b.get("captured_at"),
                "status": b.get("status"),
                "shape": b.get("shape") or "tp3-triangle",
                "image_tag": b.get("image_tag"),
                "image_id": (b.get("image_id") or "")[:19],
                "vllm_tree": (b.get("source_identity", {}).get("vllm_tree") or "")[:8],
                "vllm_patch_head": (b.get("source_identity", {}).get("vllm_patch_head") or "")[:8],
                "benchmark_reference": b.get("benchmark_reference"),
                "previous_baseline": b.get("previous_baseline"),
                "change_from_previous": b.get("change_from_previous"),
                "evidence_summary": b.get("evidence", {}).get("summary")
                if isinstance(b.get("evidence"), dict)
                else None,
                "experiments": b.get("evidence", {}).get("experiments", [])
                if isinstance(b.get("evidence"), dict)
                else [],
                "is_promoted": stem == promoted_stem,
            }
        )

    return {
        "promoted_baseline": promoted_stem,
        "promoted_short": _short_release_name(promoted_stem),
        "promoted_date": _baseline_date(promoted_stem),
        "default_shape": cluster["default_shape"],
        "container_name": cluster["container"]["name"],
        "image_tag": cluster["container"]["image"],
        "image_id": promoted_manifest["image_id"],
        "built_as": promoted_manifest.get("build", {}).get("built_as"),
        "built_utc": promoted_manifest.get("build", {}).get("built_utc"),
        "model_key": cluster["model"],
        "model_repo": src_id.get("model_repository", "deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        "model_revision": src_id.get("model_revision", "")[:8],
        "served_models": [
            "deepseek-v4.1-flash",
            cluster.get("glp", {}).get("served_model_name", "deepseek-v4.1-flash-weightless"),
        ],
        "glp": cluster.get("glp"),
        "drafts": int(cluster["drafts"]),
        "kernel_backend": cluster["kernel_backend"],
        "hosts": promoted_manifest.get("hosts", {}),
        "runtime_choices": promoted_manifest.get("runtime_choices", {}),
        "capabilities": promoted_source.get("vllm", {}).get("capabilities", []),
        "shapes": shapes_info,
        "upstreams": upstreams_summary,
        "suites": suites,
        "suites_by_path": suites_by_path,
        "timeline": timeline,
    }


def render_readme_current_baseline(data: dict[str, Any]) -> str:
    lines = [
        f"| Shape | Ranks & transport | Context | Sequences | Batch budget | KV pool / rank | Full contexts held (startup line) | Qualified reference |",
        f"| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in data["shapes"]:
        kv = s["kv_capacity"]
        ctx_label = f"{s['max_model_len']:,}"
        seq_label = f"{s['max_num_seqs']} (up to {kv['every_sequence_tokens']:,} tok each)"
        startup_tok = kv["vllm_capacity_tokens"] if s["qualified_release"] == data["promoted_short"] else kv["reported_tokens"]
        startup_ctx = kv["vllm_capacity_contexts"] if s["qualified_release"] == data["promoted_short"] else kv["reported_contexts"]
        held_label = (
            f"{kv['full_contexts']} ({kv['contexts']:.2f}x; startup `{startup_tok:,}` / `{startup_ctx:.2f}x`)"
        )
        if s["benchmark_reference"]:
            ref_target = (
                s["benchmark_reference"]
                if s["benchmark_reference"].endswith(".json")
                else f"{s['benchmark_reference']}/"
            )
            ref_link = f"[`{s['qualified_release']}`]({ref_target})"
        else:
            ref_link = "unqualified (`--unqualified-shape`)"
        default_badge = " **(default)**" if s["default"] else ""
        lines.append(
            f"| `{s['name']}`{default_badge} | TP{s['nodes']} `{s['transport']}` ({s['page_size_label']}) | "
            f"{ctx_label} | {seq_label} | {s['max_num_batched_tokens']:,} | {s['kv_cache_gib']} | "
            f"{held_label} | {ref_link} |"
        )
    return "\n".join(lines)


def render_readme_performance_summary(data: dict[str, Any]) -> str:
    shapes = {s["name"]: s for s in data["shapes"]}
    tp4_ref = data["suites_by_path"][shapes["tp4-ring"]["benchmark_reference"]]
    tp3_ref = data["suites_by_path"][shapes["tp3-triangle"]["benchmark_reference"]]

    def _prefill_map(suite_entry: dict[str, Any]) -> dict[int, float]:
        return {row["size"]: row["tps"] for row in suite_entry["prefill"]}

    p4 = _prefill_map(tp4_ref)
    p3 = _prefill_map(tp3_ref)

    c16_4 = tp4_ref["runs"].get("c16", {}).get("output_tps")
    c8_4 = tp4_ref["runs"].get("c8", {}).get("output_tps")
    c1_4 = tp4_ref["runs"].get("c1", {}).get("output_tps")
    pois_4 = tp4_ref["runs"].get("poisson", {}).get("output_tps")
    samp16_4 = tp4_ref["sampled"].get("c16", {}).get("output_tps")
    samp8_4 = tp4_ref["sampled"].get("c8", {}).get("output_tps")
    samp1_4 = tp4_ref["sampled"].get("c1", {}).get("output_tps")
    grid_4 = tp4_ref["grid"]["geomean_ms"]
    stall3_ttft = tp4_ref["stalls"].get("stall-3", {}).get("probe_ttft_p50_s")
    stall11_ttft = tp4_ref["stalls"].get("stall-11", {}).get("probe_ttft_p50_s")

    c8_3 = tp3_ref["runs"].get("c8", {}).get("output_tps")
    c4_3 = tp3_ref["runs"].get("c4", {}).get("output_tps")
    c1_3 = tp3_ref["runs"].get("c1", {}).get("output_tps")
    pois_3 = tp3_ref["runs"].get("poisson", {}).get("output_tps")
    samp16_3 = tp3_ref["sampled"].get("c16", {}).get("output_tps")
    samp8_3 = tp3_ref["sampled"].get("c8", {}).get("output_tps")
    samp1_3 = tp3_ref["sampled"].get("c1", {}).get("output_tps")
    grid_3 = tp3_ref["grid"]["geomean_ms"]
    stall3_ttft_3 = tp3_ref["stalls"].get("stall-3", {}).get("probe_ttft_p50_s")
    stall6_ttft_3 = tp3_ref["stalls"].get("stall-6", {}).get("probe_ttft_p50_s")
    tp3_stall_cell = (
        f"{stall3_ttft_3:.2f} s / {stall6_ttft_3:.2f} s"
        if stall3_ttft_3 is not None and stall6_ttft_3 is not None
        else "—"
    )

    lines = [
        "Current qualified reference runs by shape (from `config/cluster.json` and `manifests/suites/`):",
        "",
        f"| Metric | `tp4-ring` ([`{tp4_ref['release']}`]({tp4_ref['path']}/)) | `tp3-triangle` ([`{tp3_ref['release']}`]({tp3_ref['path']}/)) |",
        "| --- | ---: | ---: |",
        f"| Decode step geometric mean ({tp4_ref['grid']['shape_count']} TP4 / {tp3_ref['grid']['shape_count']} TP3 shapes) | {grid_4:.2f} ms | {grid_3:.2f} ms |",
        f"| Decode, 16 streams (`c16` output tok/s) | {c16_4:.1f} | — |",
        f"| Decode, 8 streams (`c8` output tok/s) | {c8_4:.1f} | {c8_3:.1f} |",
        f"| Decode, 4 streams (`c4` output tok/s) | — | {c4_3:.1f} |",
        f"| One conversation at a time (`single-stream` `c1` output tok/s) | {c1_4:.1f} | {c1_3:.1f} |",
        f"| Sampled decode (`temp=1.0, top_p=1.0`), c16 / c8 / c1 (tok/s) | {samp16_4:.1f} / {samp8_4:.1f} / {samp1_4:.1f} | {samp16_3:.1f} / {samp8_3:.1f} / {samp1_3:.1f} |",
        f"| Poisson arrivals (0.7 TP4 / 0.42 TP3 req/s), output (tok/s) | {pois_4:.1f} | {pois_3:.1f} |",
        f"| Source-text prefill, 32K (tok/s) | {p4[32768]:,.0f} | {p3[32768]:,.0f} |",
        f"| Prefill, 131K (tok/s) | {p4[131072]:,.0f} | {p3[131072]:,.0f} |",
        f"| Prefill, 262K (tok/s) | {p4[262144]:,.0f} | {p3[262144]:,.0f} |",
        f"| Prefill, 524K (TP4) / 500K (TP3) (tok/s) | {p4[524288]:,.0f} | {p3.get(524288, p3.get(500000)):,.0f} |",
        f"| Prefill, 1M (tok/s) | {p4[1000000]:,.0f} | {f'{p3[1000000]:,.0f}' if 1000000 in p3 else '3,952 (cold 1M)'} |",
        f"| Short-request TTFT p50 during 131K prefill, 3 / 11 (TP4) or 3 / 6 (TP3) streams | {stall3_ttft:.2f} s / {stall11_ttft:.2f} s | {tp3_stall_cell} |",
    ]
    return "\n".join(lines)


def render_current_state_active_settings(data: dict[str, Any]) -> str:
    shapes = {s["name"]: s for s in data["shapes"]}
    tp4 = shapes["tp4-ring"]
    tp3 = shapes["tp3-triangle"]
    tp3_4k = shapes["tp3-triangle-4k"]
    kv4 = tp4["kv_capacity"]
    kv3 = tp3["kv_capacity"]
    kv3_4k = tp3_4k["kv_capacity"]

    up_map = {u["name"]: u for u in data["upstreams"]}
    vllm_u = up_map["vllm"]
    tl_u = up_map["tilelang"]
    tk_u = up_map["tile_kernels"]
    sn_u = up_map["sparknet"]
    nccl_u = up_map["nccl"]
    hosts = data["hosts"]
    ms = hosts.get("memory_saver", {})

    rows = [
        (
            "Promoted baseline",
            f"`{data['promoted_baseline']}` (`{data['promoted_short']}`, frozen in `releases/{data['promoted_short']}/`); default shape `{data['default_shape']}`",
        ),
        (
            "Served models",
            f"`{data['served_models'][0]}` (`{data['model_repo']}`, revision `{data['model_revision']}`) and per-request GLP-steered `{data['served_models'][1]}` (`{data['glp']['repo']}`, revision `{data['glp']['revision'][:8]}`)",
        ),
        (
            "Kernel backend",
            f"TileLang only (`kernel_backend: {data['kernel_backend']}`, vLLM 0493-0496): TileLang {tl_u['version']} (`{tl_u['tree'][:8]}`), DeepSeek TileKernels {tk_u['version']} (`{tk_u['tree'][:8]}`), sparknet {sn_u['version']} (`{sn_u['tree'][:8]}`) one-shot RoCE collectives; every serving kernel is TileLang including the vision tower, CED row selection, RoPE, compressor, KV writers, Engram gate/hash/gather, sparse MLA, WO projection, drafter context-KV, sampling and block verification",
        ),
        (
            "Sources",
            f"Local Inference Lab `integration/karmic-kraken-beta` vLLM `{vllm_u['revision'][:8]}` + `{vllm_u['patch_series']}` ({vllm_u['patch_count']} patches, patch head `{vllm_u['patch_head'][:8]}`, tree `{vllm_u['tree'][:8]}`), TileLang {tl_u['version']} ({tl_u['patch_count']} patches), TileKernels {tk_u['version']} ({tk_u['patch_count']} patch), sparknet {sn_u['version']}, NCCL 2.30.7 ({nccl_u['patch_count']} patches, tree `{nccl_u['tree'][:8]}`); direct checkpoint loader (`--load-format direct`, vLLM 0499, `VLLM_PLUGINS` empty)",
        ),
        (
            "Image",
            f"`{data['image_tag']}` (built as `{data['built_as']}`, `{data['image_id'][:19]}`), one digest on all ranks, built by `bin/spark build`",
        ),
        (
            "Decode tiles & PDL",
            "block-32 FP8 decode rows on 16-, 32- or 64-row tiles by row count and split-K tiles up to 128 rows for 16-stream steps, programmatic dependent launch (PDL) along the decode chain (vLLM 0520-0525), attention front in one launch (0543-0544), 64-row prefill MoE tiles and panel rasterization past L2 (0043)",
        ),
        (
            "Batch invariance",
            f"temperature-0 outputs do not depend on the batch: rank-order FP32 reductions with one rounding, compressor projection one TileLang arithmetic at every row count, aligned prefill chunks (`--long-prefill-token-threshold` {tp4['long_prefill_token_threshold']:,} on `tp4-ring`, {tp3['long_prefill_token_threshold']:,} on `tp3-triangle`, {tp3_4k['long_prefill_token_threshold']:,} on `tp3-triangle-4k`), CED replays a prompt's last 128 rows wherever its chunks end (0526-0527), and mixed steps run sparse MLA per request",
        ),
        (
            "Hosts",
            f"{hosts.get('os', 'DGX Spark 26.09.2')}, kernel `{hosts.get('kernel', '7.0.0-1019-nvidia-64k with kho=off')}`, driver {hosts.get('nvidia_driver', '580.178.04')}, `{hosts.get('boot_target', 'multi-user.target (no desktop)')}`, `--cpuset-cpus=5-9,15-19`, glibc malloc pinned (`MALLOC_ARENA_MAX=2`, `MALLOC_MMAP_THRESHOLD_=1048576`)",
        ),
        (
            "Tensor parallel ranks & transport",
            f"`tp4-ring`: 4 ranks (`{tp4['transport']}`); `tp3-triangle`: 3 ranks (`{tp3['transport']}`); `tp3-triangle-4k`: 3 ranks (`{tp3_4k['transport']}`)",
        ),
        (
            "Maximum model length",
            f"`tp4-ring` and `tp3-triangle`: {tp4['max_model_len']:,} tokens; `tp3-triangle-4k`: {tp3_4k['max_model_len']:,} tokens",
        ),
        (
            "Maximum sequences",
            f"`tp4-ring`: {tp4['max_num_seqs']}; `tp3-triangle` and `tp3-triangle-4k`: {tp3['max_num_seqs']}",
        ),
        ("Maximum parallel prefills", "1 (`--max-parallel-prefills 1`)"),
        (
            "Batched-token budget",
            f"`tp4-ring`: {tp4['max_num_batched_tokens']:,}; `tp3-triangle`: {tp3['max_num_batched_tokens']:,} (qualified on `spark-ds41f-1030`); `tp3-triangle-4k`: {tp3_4k['max_num_batched_tokens']:,}",
        ),
        (
            "Prefill yield & late admission",
            "prefill yield enabled (`--prefill-yield short`, `--prefill-yield-limit-s 10`, vLLM 0545) so waiting short requests take the next step from a long prefill; late admission at decoder layer 30 (`VLLM_LATE_ADMISSION_LAYER=30`, vLLM 0222, 0226)",
        ),
        (
            "Prefill sequence parallelism",
            "from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers with non-final row skip (vLLM 0014-0018, 0229-0255)",
        ),
        (
            "Indexer under sequence parallelism",
            "each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025), ties broken by lowest logical position in TileLang",
        ),
        (
            "Explicit KV memory",
            f"`tp4-ring`: {tp4['kv_cache_gib']} ({tp4['kv_cache_memory_bytes']:,} B) per rank; `tp3-triangle`: {tp3['kv_cache_gib']} ({tp3['kv_cache_memory_bytes']:,} B) per rank; `tp3-triangle-4k`: {tp3_4k['kv_cache_gib']} ({tp3_4k['kv_cache_memory_bytes']:,} B) per rank; no per-allocation KV zeroing (vLLM 0502-0503)",
        ),
        (
            "KV capacity",
            f"`tp4-ring`: {kv4['vllm_capacity_tokens']:,} startup tokens ({kv4['vllm_capacity_contexts']:.2f}x; holds {kv4['full_contexts']} full 1M contexts or {tp4['max_num_seqs']} x {kv4['every_sequence_tokens']:,}); `tp3-triangle`: {kv3['vllm_capacity_tokens']:,} startup tokens ({kv3['vllm_capacity_contexts']:.2f}x; holds {kv3['full_contexts']} full 1M contexts or {tp3['max_num_seqs']} x {kv3['every_sequence_tokens']:,}); `tp3-triangle-4k`: holds {kv3_4k['full_contexts']} full 256K contexts ([KV capacity](memory-profiles.md#kv-capacity))",
        ),
        (
            "Expert scale codes",
            "routed experts' UE8M0 weight scales as a per-row base exponent and 4-bit deltas (vLLM 0182-0183), with 2-bit deltas for narrow experts stored in the 4-bit codes' buffer (0563-0564, 0566; model loading 71.67 GiB a rank at TP4)",
        ),
        (
            "Memory-saver",
            f"signed DKMS {ms.get('version', '0.4.1')} (`{ms.get('package', 'dgx-spark-memory-saver')}`) on all nodes; loaded UVM leaf-table packing enabled (`uvm_leaf_packing: {hosts.get('uvm_leaf_packing', 'Y')}`)",
        ),
        (
            "Page-size profiles",
            f"64 KiB selected (`tp4-ring`, `tp3-triangle`); 4 KiB (`tp3-triangle-4k`) retains {tp3_4k['kv_cache_gib']} KV and a {tp3_4k['max_model_len']:,}-token limit",
        ),
        (
            "Output head & display carve-out",
            "exact 12-bit packed BF16 (`VLLM_DS41_PACKED_BF16_LM_HEAD=1`), bit-identical to the checkpoint, with FP32 logits (`VLLM_DS41_HEAD_DTYPE=float32`); kept in regular memory on `tp4-ring` (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0`) and in the GB10 display carve-out on `tp3-triangle` (`=1`)",
        ),
        (
            "Reference precision",
            "DeepSeek's `inference/model.py` term for term: FP32 logits, exact FP32 mHC stream projection (three TF32 parts per weight), shared expert SwiGLU in FP32 with one rounding, MoE partials rounded once per rank, and WO-A's input through `act_quant` (E4M3, UE8M0 per 32, vLLM 0177, 0179)",
        ),
        (
            "Image input",
            "TileLang vision tower loaded (`--limit-mm-per-prompt '{\"image\":4}'`, `VLLM_MAX_IMAGE_PIXELS=16777216`, `--mm-processor-cache-gb 0`; [below](#images-per-request))",
        ),
        (
            "DSpark speculative decoding",
            f"{data['drafts']} draft tokens, draft TP = node count, fixed linear step cost budget (`adaptive_verification_row_cost: 0.035`, vLLM 0184, nothing timed at boot), dead verification rows below survival 0.2 (`0504`), block rejection with fresh draws per round (`0228`), sink draft-zero rejection on by default (`0221, 0223`), vocabulary-parallel greedy drafts and shard argmax verification (`0112-0114`)",
        ),
        (
            "Collectives",
            f"sparknet {sn_u['version']} one-shot RoCE all-reduce (up to 2 MiB) and all-gather (`oneshot-ring4` on TP4, `oneshot-direct` on TP3), TP broadcast on one-shot all-gather (`0185`), GPU-clock timeout 10 s (`SPARKNET_ROCE_TIMEOUT_MS=10000`), no fallback (`0142`)",
        ),
        (
            "CUDA graphs & warmup",
            f"`FULL_AND_PIECEWISE`, capture sizes 1-{tp4['max_cudagraph_capture_size']} on `tp4-ring` (1-{tp3['max_cudagraph_capture_size']} on `tp3-triangle`); all declared TileLang calls prepared and cache frozen before memory profiling (`--jit-monitor-mode off`)",
        ),
        (
            "Engram projection & rows",
            "sharded across ranks (`projection_tp`); one native reader thread reads both tables in layer order from interleaved whole hash columns; host hashes prefill tokens a step early except drafted-prefill requests, which the GPU hashes (vLLM 0145-0154, 0180, 0567)",
        ),
        (
            "Memory guards",
            "5 GiB startup (0.25 s), 3 GiB steady (2 s), each killing on its first sample below; hosts set `vm.watermark_boost_factor=0`",
        ),
        (
            "Prefix caching & API semantics",
            "256-token blocks (`--enable-prefix-caching`, `--enable-prompt-tokens-details`); thinking enabled by default, top-level `thinking` switch, effort aliases, `response_format`, `tool_choice: none`, assistant prefix, and generation defaults `temperature: 1.0, top_p: 1.0`",
        ),
    ]

    lines = [
        "| Setting | Active value |",
        "| --- | --- |",
    ]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def render_todo_next_release_status(data: dict[str, Any]) -> str:
    shapes = {s["name"]: s for s in data["shapes"]}
    tp4 = shapes["tp4-ring"]
    tp3 = shapes["tp3-triangle"]
    return (
        f"Production is **{data['promoted_short']}** (`manifests/baselines/{data['promoted_baseline']}.json`, "
        f"image `{data['image_tag']}`), promoted {data['promoted_date']} on `tp4-ring` "
        f"([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md), "
        f"[suite reference]({tp4['benchmark_reference']}/)). "
        f"`tp3-triangle`'s current qualified reference is **{tp3['qualified_release']}** "
        f"([suite reference]({tp3['benchmark_reference']}/)); qualifying `{data['promoted_short']}`'s "
        f"image on the triangle at its {tp3['max_num_batched_tokens']:,}-token prefill budget is queued "
        f"once the three-node triangle is cabled (#668)."
    )


def render_todo_current_reference(data: dict[str, Any]) -> str:
    shapes = {s["name"]: s for s in data["shapes"]}
    tp4_ref = data["suites_by_path"][shapes["tp4-ring"]["benchmark_reference"]]
    tp3_ref = data["suites_by_path"][shapes["tp3-triangle"]["benchmark_reference"]]
    p4 = {row["size"]: row["tps"] for row in tp4_ref["prefill"]}
    p3 = {row["size"]: row["tps"] for row in tp3_ref["prefill"]}

    lines = [
        f"Active baseline manifest: [`manifests/baselines/{data['promoted_baseline']}.json`](manifests/baselines/{data['promoted_baseline']}.json) (`{data['promoted_short']}`). "
        "Releases are judged by the servebench release, single-stream and sampled suites (`docs/release-benchmark.md`, `scripts/servebench_suite_table.py`). "
        f"Current benchmark references are [`{tp4_ref['path']}`]({tp4_ref['path']}/) for `tp4-ring` and [`{tp3_ref['path']}`]({tp3_ref['path']}/) for `tp3-triangle`:",
        "",
        f"| Measure | `tp4-ring` (`{tp4_ref['release']}`, 1M, 16 seqs, 16K batch) | `tp3-triangle` (`{tp3_ref['release']}`, 1M, 8 seqs, 8K/6K batch) |",
        "| --- | ---: | ---: |",
        f"| Decode grid geometric mean ({tp4_ref['grid']['shape_count']} TP4 / {tp3_ref['grid']['shape_count']} TP3 shapes) | {tp4_ref['grid']['geomean_ms']:.2f} ms | {tp3_ref['grid']['geomean_ms']:.2f} ms |",
        f"| Decode `c16`, 16 streams (output tok/s) | {tp4_ref['runs']['c16']['output_tps']:.1f} | — |",
        f"| Decode `c8`, 8 streams (output tok/s) | {tp4_ref['runs']['c8']['output_tps']:.1f} | {tp3_ref['runs']['c8']['output_tps']:.1f} |",
        f"| Decode `c4`, 4 streams (output tok/s) | — | {tp3_ref['runs']['c4']['output_tps']:.1f} |",
        f"| Single-stream `c1` (output tok/s) | {tp4_ref['runs']['c1']['output_tps']:.1f} | {tp3_ref['runs']['c1']['output_tps']:.1f} |",
        f"| Sampled `c16` / `c8` / `c1` (`temp=1.0`, output tok/s) | {tp4_ref['sampled']['c16']['output_tps']:.1f} / {tp4_ref['sampled']['c8']['output_tps']:.1f} / {tp4_ref['sampled']['c1']['output_tps']:.1f} | {tp3_ref['sampled']['c16']['output_tps']:.1f} / {tp3_ref['sampled']['c8']['output_tps']:.1f} / {tp3_ref['sampled']['c1']['output_tps']:.1f} |",
        f"| Poisson arrivals (0.7 TP4 / 0.42 TP3 req/s, output tok/s) | {tp4_ref['runs']['poisson']['output_tps']:.1f} | {tp3_ref['runs']['poisson']['output_tps']:.1f} |",
        f"| Source-text prefill 32K (tok/s) | {p4[32768]:,.0f} | {p3[32768]:,.0f} |",
        f"| Prefill 131K (tok/s) | {p4[131072]:,.0f} | {p3[131072]:,.0f} |",
        f"| Prefill 262K (tok/s) | {p4[262144]:,.0f} | {p3[262144]:,.0f} |",
        f"| Prefill 524K (TP4) / 500K (TP3) (tok/s) | {p4[524288]:,.0f} | {p3.get(524288, p3.get(500000)):,.0f} |",
        f"| Prefill 1M (tok/s) | {p4[1000000]:,.0f} | {f'{p3[1000000]:,.0f}' if 1000000 in p3 else '3,952'} |",
        f"| Short-request TTFT p50 during 131K prefill (3 / 11 streams) | {tp4_ref['stalls']['stall-3']['probe_ttft_p50_s']:.2f} s / {tp4_ref['stalls']['stall-11']['probe_ttft_p50_s']:.2f} s | — |",
    ]
    return "\n".join(lines)


def _replace_markers(text: str, blocks: dict[str, str], path_label: str) -> str:
    seen: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        begin, key, end = match.group(1), match.group(2), match.group(3)
        if key not in blocks:
            raise KeyError(f"{path_label}: unknown generated block {key!r}")
        seen.add(key)
        return f"{begin}\n{blocks[key]}\n{end}"

    updated = MARKER_RE.sub(_sub, text)
    missing = set(blocks) - seen
    if missing:
        raise RuntimeError(f"{path_label}: missing markers for {', '.join(sorted(missing))}")
    return updated


def render_site_html(data: dict[str, Any]) -> str:
    # Build a compact, deterministic JSON payload for the interactive page.
    payload = {
        "promoted_baseline": data["promoted_baseline"],
        "promoted_short": data["promoted_short"],
        "promoted_date": data["promoted_date"],
        "default_shape": data["default_shape"],
        "container_name": data["container_name"],
        "image_tag": data["image_tag"],
        "image_id": data["image_id"],
        "built_as": data["built_as"],
        "built_utc": data["built_utc"],
        "model_repo": data["model_repo"],
        "model_revision": data["model_revision"],
        "served_models": data["served_models"],
        "glp": data["glp"],
        "drafts": data["drafts"],
        "kernel_backend": data["kernel_backend"],
        "hosts": data["hosts"],
        "runtime_choices": data["runtime_choices"],
        "capabilities": data["capabilities"],
        "shapes": data["shapes"],
        "upstreams": data["upstreams"],
        "suites": data["suites"],
        "timeline": data["timeline"],
    }
    embedded_json = json.dumps(payload, indent=2, sort_keys=True)
    title = f"spark-ds41f — {data['promoted_short']} Manifest & Benchmark Observatory"

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --bg-elev: #111827;
      --bg-card: #161f30;
      --bg-hover: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --border: #263248;
      --accent: #10b981;
      --accent-soft: rgba(16, 185, 129, 0.14);
      --cyan: #06b6d4;
      --cyan-soft: rgba(6, 182, 212, 0.14);
      --amber: #f59e0b;
      --amber-soft: rgba(245, 158, 11, 0.14);
      --indigo: #818cf8;
      --rose: #f43f5e;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
    }}
    html[data-theme="light"] {{
      --bg: #f8fafc;
      --bg-elev: #ffffff;
      --bg-card: #ffffff;
      --bg-hover: #f1f5f9;
      --text: #0f172a;
      --text-muted: #475569;
      --border: #e2e8f0;
      --accent: #059669;
      --accent-soft: rgba(5, 150, 105, 0.10);
      --cyan: #0891b2;
      --cyan-soft: rgba(8, 145, 178, 0.10);
      --amber: #d97706;
      --amber-soft: rgba(217, 119, 6, 0.10);
      --indigo: #4f46e5;
      --rose: #e11d48;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.5;
      font-variant-numeric: tabular-nums;
    }}
    a {{ color: var(--cyan); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .container {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px 20px 72px;
    }}
    header.hero {{
      background: linear-gradient(135deg, var(--bg-elev) 0%, var(--bg-card) 100%);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px 28px;
      margin-bottom: 24px;
      position: relative;
    }}
    .hero-top {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      font-family: var(--mono);
      border: 1px solid var(--border);
      background: var(--bg);
    }}
    .badge.live {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: var(--accent);
    }}
    .badge.cyan {{
      background: var(--cyan-soft);
      color: var(--cyan);
      border-color: var(--cyan);
    }}
    .badge.amber {{
      background: var(--amber-soft);
      color: var(--amber);
      border-color: var(--amber);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--text-muted);
      font-size: 14.5px;
      max-width: 880px;
    }}
    .hero-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    button.btn, select.select, input.search {{
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 12px;
      font-size: 13px;
      font-family: var(--sans);
      cursor: pointer;
    }}
    button.btn:hover, select.select:hover {{
      border-color: var(--cyan);
    }}
    button.btn.active {{
      background: var(--cyan-soft);
      color: var(--cyan);
      border-color: var(--cyan);
      font-weight: 600;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .kpi-card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .kpi-label {{
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}
    .kpi-value {{
      font-size: 23px;
      font-weight: 700;
      font-family: var(--mono);
      color: var(--text);
    }}
    .kpi-delta {{
      font-size: 12px;
      font-family: var(--mono);
      color: var(--accent);
      margin-top: 3px;
    }}
    nav.tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }}
    .section {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px 24px;
      margin-bottom: 22px;
    }}
    .section-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .section h2 {{
      margin: 0;
      font-size: 19px;
      letter-spacing: -0.01em;
    }}
    .section p.desc {{
      margin: 4px 0 0;
      color: var(--text-muted);
      font-size: 13.5px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    @media (max-width: 640px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}
    .chart-box {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .chart-title {{
      font-size: 13.5px;
      font-weight: 600;
      margin: 0 0 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .chart-title span.sub {{
      font-weight: 400;
      font-size: 12px;
      color: var(--text-muted);
      font-family: var(--mono);
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--bg);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{
      background: var(--bg-elev);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    td.num, th.num {{
      text-align: right;
      font-family: var(--mono);
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: var(--bg-hover); }}
    code {{
      font-family: var(--mono);
      font-size: 12px;
      background: var(--bg-elev);
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }}
    .verdict-better {{
      color: var(--accent);
      font-weight: 600;
      font-family: var(--mono);
    }}
    .verdict-same {{
      color: var(--text-muted);
      font-family: var(--mono);
    }}
    .verdict-worse {{
      color: var(--rose);
      font-weight: 600;
      font-family: var(--mono);
    }}
    .shape-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .shape-card {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
    }}
    .shape-card.default {{
      border-color: var(--accent);
    }}
    .kv-bar {{
      height: 10px;
      border-radius: 999px;
      background: var(--bg-elev);
      overflow: hidden;
      margin: 10px 0;
      border: 1px solid var(--border);
    }}
    .kv-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--cyan));
    }}
    .timeline-list {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .timeline-item {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
    }}
    .timeline-item.promoted {{
      border-color: var(--accent);
    }}
    .timeline-head {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }}
    .cap-tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
    }}
    .cap-tag {{
      font-family: var(--mono);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 6px;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }}
    footer {{
      text-align: center;
      color: var(--text-muted);
      font-size: 12.5px;
      margin-top: 36px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <div class="hero-top">
        <div>
          <div class="badge-row" id="hero-badges"></div>
          <h1 id="hero-title">spark-ds41f</h1>
          <p class="subtitle" id="hero-subtitle"></p>
        </div>
        <div class="hero-actions">
          <a class="btn" href="https://github.com/christopherowen/spark-ds41f" target="_blank" rel="noopener">GitHub Repo</a>
          <button class="btn" id="theme-toggle" type="button">Light / Dark</button>
        </div>
      </div>
      <div class="kpi-grid" id="hero-kpis"></div>
    </header>

    <section class="section" id="sec-performance">
      <div class="section-header">
        <div>
          <h2>Release-over-Release Performance Observatory</h2>
          <p class="desc">Driven directly from <code>manifests/suites/*/</code> across all recorded servebench release, single-stream, and sampled runs.</p>
        </div>
        <div class="controls">
          <label style="font-size:12.5px;color:var(--text-muted)">Topology:</label>
          <select class="select" id="topo-select">
            <option value="TP4">TP4 Ring (4× DGX Spark)</option>
            <option value="TP3">TP3 Triangle (3× DGX Spark)</option>
          </select>
          <label style="font-size:12.5px;color:var(--text-muted)">Base:</label>
          <select class="select" id="base-suite-select"></select>
          <label style="font-size:12.5px;color:var(--text-muted)">Candidate:</label>
          <select class="select" id="cand-suite-select"></select>
        </div>
      </div>

      <div class="chart-grid">
        <div class="chart-box">
          <div class="chart-title">
            <span>Source-Text Prefill Throughput (32K → 1M)</span>
            <span class="sub">tok/s (higher is better)</span>
          </div>
          <div id="chart-prefill"></div>
        </div>
        <div class="chart-box">
          <div class="chart-title">
            <span>Decode Throughput across Releases</span>
            <span class="sub">output tok/s (higher is better)</span>
          </div>
          <div id="chart-decode"></div>
        </div>
      </div>

      <div class="chart-grid">
        <div class="chart-box">
          <div class="chart-title">
            <span>Engine Decode Step-Time Grid (Base vs Candidate)</span>
            <span class="sub">trimmed mean ms / step (lower is better)</span>
          </div>
          <div id="chart-grid"></div>
        </div>
        <div class="chart-box">
          <div class="chart-title">
            <span>DSpark Speculative Survival &amp; Prefill Yield</span>
            <span class="sub">draft positions k=0..4 &amp; 131K stall probes</span>
          </div>
          <div id="chart-spec-stall"></div>
        </div>
      </div>

      <div class="table-wrap">
        <table id="suite-compare-table">
          <thead>
            <tr>
              <th>Workload / Metric</th>
              <th class="num" id="th-base">Base</th>
              <th class="num" id="th-cand">Candidate</th>
              <th class="num">Delta</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody id="suite-compare-tbody"></tbody>
        </table>
      </div>
    </section>

    <section class="section" id="sec-shapes">
      <div class="section-header">
        <div>
          <h2>Cluster Shapes &amp; KV Pool Sizing</h2>
          <p class="desc">Resolved from <code>config/cluster.json</code> via <code>scripts/recipe.py</code> and <code>scripts/kv_capacity.py</code> (227,840-byte pool blocks).</p>
        </div>
      </div>
      <div class="shape-cards" id="shape-cards"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Shape</th>
              <th>Ranks &amp; Transport</th>
              <th>Kernel / Pages</th>
              <th class="num">Context Limit</th>
              <th class="num">Max Seqs</th>
              <th class="num">Batch Tokens</th>
              <th class="num">KV Pool / Rank</th>
              <th class="num">Pool Blocks</th>
              <th class="num">Full Contexts Held</th>
              <th class="num">vLLM 0491 Startup Line</th>
              <th>Qualified Suite</th>
            </tr>
          </thead>
          <tbody id="shapes-tbody"></tbody>
        </table>
      </div>
    </section>

    <section class="section" id="sec-upstreams">
      <div class="section-header">
        <div>
          <h2>Pinned Upstreams, Patch Stacks &amp; Active Runtime Choices</h2>
          <p class="desc">Synchronized from <code>upstreams.lock.json</code>, <code>manifests/baselines/{html.escape(data['promoted_baseline'])}.json</code>, and <code>manifests/sources/</code>.</p>
        </div>
        <div class="controls">
          <input class="search" id="choice-search" type="search" placeholder="Filter runtime choices or capabilities...">
        </div>
      </div>
      <div class="table-wrap" style="margin-bottom:18px">
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Version / Ref</th>
              <th class="num">Pinned Revision</th>
              <th class="num">Patch Series</th>
              <th class="num">Verified Tree</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody id="upstreams-tbody"></tbody>
        </table>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width:220px">Runtime Choice</th>
              <th>Active Specification in <code>{html.escape(data['promoted_short'])}</code></th>
            </tr>
          </thead>
          <tbody id="choices-tbody"></tbody>
        </table>
      </div>
      <div class="cap-tags" id="cap-tags"></div>
    </section>

    <section class="section" id="sec-timeline">
      <div class="section-header">
        <div>
          <h2>Baseline Manifest History (35 Releases)</h2>
          <p class="desc">Complete immutable promotion history from <code>manifests/baselines/*.json</code>.</p>
        </div>
        <div class="controls">
          <input class="search" id="timeline-search" type="search" placeholder="Search releases (e.g. 1030, r6x, engram, glp)...">
        </div>
      </div>
      <div class="timeline-list" id="timeline-list"></div>
    </section>

    <footer>
      Generated deterministically from repository manifests by <code>scripts/render_docs.py</code> &bull; Verified in CI by <code>tests/test_releases.py</code>
    </footer>
  </div>

  <script id="manifest-data" type="application/json">
{embedded_json}
  </script>
  <script>
    (function () {{
      const DATA = JSON.parse(document.getElementById("manifest-data").textContent);

      // Theme toggle
      const htmlEl = document.documentElement;
      document.getElementById("theme-toggle").addEventListener("click", () => {{
        htmlEl.setAttribute("data-theme", htmlEl.getAttribute("data-theme") === "dark" ? "light" : "dark");
      }});

      function esc(str) {{
        if (str === null || str === undefined) return "";
        return String(str)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;");
      }}

      // Render Hero
      const tp4Shape = DATA.shapes.find(s => s.name === "tp4-ring") || DATA.shapes[0];
      const tp3Shape = DATA.shapes.find(s => s.name === "tp3-triangle") || DATA.shapes[1];
      const tp4Suite = DATA.suites.find(s => s.path === tp4Shape.benchmark_reference);
      const tp3Suite = DATA.suites.find(s => s.path === tp3Shape.benchmark_reference);

      document.getElementById("hero-badges").innerHTML = [
        `<span class="badge live">● PRODUCTION: ${{esc(DATA.promoted_short)}}</span>`,
        `<span class="badge cyan">Image: ${{esc(DATA.image_tag)}} (${{esc(DATA.image_id.slice(0, 19))}})</span>`,
        `<span class="badge cyan">Model: ${{esc(DATA.model_repo)}} (${{esc(DATA.model_revision)}})</span>`,
        `<span class="badge amber">GLP Dual-Serve: ${{esc(DATA.served_models[1])}}</span>`,
        `<span class="badge">Backend: TileLang-only + sparknet 0.5.0</span>`
      ].join("");

      document.getElementById("hero-title").textContent =
        `spark-ds41f — ${{DATA.promoted_short}} Manifest & Performance Observatory`;
      document.getElementById("hero-subtitle").textContent =
        `Reproducible Docker/vLLM deployment for DeepSeek-V4.1-Flash-1017 at 1,048,576 context on switchless 4-node ring (TP4) and 3-node triangle (TP3) DGX Spark fabrics. Driven directly from config/cluster.json, upstreams.lock.json, and manifests/.`;

      const p32_4 = (tp4Suite.prefill.find(p => p.size === 32768) || {{}}).tps || 0;
      const p131_4 = (tp4Suite.prefill.find(p => p.size === 131072) || {{}}).tps || 0;
      const p1m_4 = (tp4Suite.prefill.find(p => p.size === 1000000) || {{}}).tps || 0;
      const stall3 = (tp4Suite.stalls["stall-3"] || {{}}).probe_ttft_p50_s;

      const kpis = [
        {{
          label: "TP4 Decode (16 Streams)",
          value: `${{tp4Suite.runs.c16.output_tps.toFixed(1)}} tok/s`,
          delta: `+2.40% vs 1025 (paired +2.05%)`
        }},
        {{
          label: "TP4 Single-Stream (c1)",
          value: `${{tp4Suite.runs.c1.output_tps.toFixed(1)}} tok/s`,
          delta: `78.3 paired (+3.61%) · TTFT 0.121s`
        }},
        {{
          label: "TP4 Prefill (131K / 1M)",
          value: `${{Math.round(p131_4).toLocaleString()}} / ${{Math.round(p1m_4).toLocaleString()}}`,
          delta: `tok/s (+8.8% at 131K, +12.2% at 1M)`
        }},
        {{
          label: "TP4 Decode Grid (20 Shapes)",
          value: `${{tp4Suite.grid.geomean_ms.toFixed(2)}} ms`,
          delta: `-4.62% ±0.15% step geomean vs 1025`
        }},
        {{
          label: "Prefill Yield Short TTFT",
          value: `${{stall3 ? stall3.toFixed(2) + " s" : "—"}}`,
          delta: `3.7× sooner beside 131K prefill (was 12.36s)`
        }},
        {{
          label: "KV Pool Capacity (TP4 / TP3)",
          value: `${{tp4Shape.kv_capacity.vllm_capacity_contexts.toFixed(2)}}× / ${{tp3Shape.kv_capacity.contexts.toFixed(2)}}×`,
          delta: `11 full 1M contexts on TP4 · 5 on TP3`
        }}
      ];
      document.getElementById("hero-kpis").innerHTML = kpis.map(k => `
        <div class="kpi-card">
          <div class="kpi-label">${{esc(k.label)}}</div>
          <div class="kpi-value">${{esc(k.value)}}</div>
          <div class="kpi-delta">${{esc(k.delta)}}</div>
        </div>
      `).join("");

      // Performance Observatory Controls
      const topoSelect = document.getElementById("topo-select");
      const baseSelect = document.getElementById("base-suite-select");
      const candSelect = document.getElementById("cand-suite-select");

      function populateSuiteSelectors() {{
        const topo = topoSelect.value;
        const matching = DATA.suites.filter(s => s.topology === topo);
        baseSelect.innerHTML = matching.map((s, idx) =>
          `<option value="${{esc(s.id)}}">${{esc(s.release)}} (${{esc(s.date)}})</option>`
        ).join("");
        candSelect.innerHTML = matching.map((s, idx) =>
          `<option value="${{esc(s.id)}}">${{esc(s.release)}} (${{esc(s.date)}})</option>`
        ).join("");
        if (matching.length >= 2) {{
          baseSelect.selectedIndex = matching.length - 2;
          candSelect.selectedIndex = matching.length - 1;
        }} else if (matching.length === 1) {{
          baseSelect.selectedIndex = 0;
          candSelect.selectedIndex = 0;
        }}
        renderChartsAndTable();
      }}

      function pctDelta(baseVal, candVal, lowerBetter = false) {{
        if (!baseVal || !candVal) return null;
        const pct = ((candVal - baseVal) / baseVal) * 100;
        const good = lowerBetter ? pct < -0.3 : pct > 0.3;
        const bad = lowerBetter ? pct > 0.3 : pct < -0.3;
        const cls = good ? "verdict-better" : (bad ? "verdict-worse" : "verdict-same");
        const sign = pct > 0 ? "+" : "";
        return `<span class="${{cls}}">${{sign}}${{pct.toFixed(2)}}%</span>`;
      }}

      function renderPrefillSvg(suites, baseSuite, candSuite) {{
        const w = 520, h = 230, padL = 52, padR = 18, padT = 18, padB = 34;
        const sizes = [32768, 131072, 262144, 524288, 1000000];
        const sizeLabels = ["32K", "131K", "262K", "500K/524K", "1M"];
        const allVals = [];
        suites.forEach(s => s.prefill.forEach(p => allVals.push(p.tps)));
        const maxV = Math.max(8000, ...allVals) * 1.05;
        const minV = Math.max(0, Math.min(3000, ...allVals) * 0.9);

        const xFor = i => padL + (i / (sizes.length - 1)) * (w - padL - padR);
        const yFor = v => padT + (1 - (v - minV) / (maxV - minV)) * (h - padT - padB);

        let gridLines = "";
        for (let g = 3500; g <= 7500; g += 1000) {{
          if (g < minV || g > maxV) continue;
          const y = yFor(g);
          gridLines += `<line x1="${{padL}}" y1="${{y}}" x2="${{w - padR}}" y2="${{y}}" stroke="var(--border)" stroke-dasharray="3,3"/>`;
          gridLines += `<text x="${{padL - 6}}" y="${{y + 4}}" text-anchor="end" fill="var(--text-muted)" font-size="10.5" font-family="var(--mono)">${{(g/1000).toFixed(1)}}k</text>`;
        }}
        sizeLabels.forEach((lbl, i) => {{
          const x = xFor(i);
          gridLines += `<text x="${{x}}" y="${{h - 10}}" text-anchor="middle" fill="var(--text-muted)" font-size="11" font-family="var(--mono)">${{lbl}}</text>`;
        }});

        const palette = ["#64748b", "#94a3b8", "#818cf8", "#f59e0b", "#06b6d4", "#10b981"];
        let paths = "";
        suites.forEach((s, idx) => {{
          const isCand = s.id === candSuite.id;
          const isBase = s.id === baseSuite.id;
          const color = isCand ? "var(--accent)" : (isBase ? "var(--cyan)" : palette[idx % palette.length]);
          const opacity = (isCand || isBase) ? 1.0 : 0.40;
          const width = isCand ? 3.0 : (isBase ? 2.4 : 1.4);
          const pts = [];
          sizes.forEach((sz, i) => {{
            const row = s.prefill.find(p => p.size === sz || (sz === 524288 && p.size === 500000));
            if (row) pts.push({{ x: xFor(i), y: yFor(row.tps), tps: row.tps }});
          }});
          if (pts.length > 1) {{
            const d = pts.map((p, i) => `${{i === 0 ? "M" : "L"}}${{p.x.toFixed(1)}},${{p.y.toFixed(1)}}`).join(" ");
            paths += `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="${{width}}" opacity="${{opacity}}"/>`;
            pts.forEach(p => {{
              paths += `<circle cx="${{p.x.toFixed(1)}}" cy="${{p.y.toFixed(1)}}" r="${{isCand ? 4 : 3}}" fill="${{color}}" opacity="${{opacity}}"><title>${{esc(s.release)}}: ${{Math.round(p.tps).toLocaleString()}} tok/s</title></circle>`;
            }});
          }}
        }});

        return `<svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="230">${{gridLines}}${{paths}}</svg>`;
      }}

      function renderDecodeSvg(suites, baseSuite, candSuite) {{
        const w = 520, h = 230, padL = 46, padR = 18, padT = 20, padB = 42;
        const maxV = 280;
        const yFor = v => padT + (1 - v / maxV) * (h - padT - padB);
        let svg = "";
        for (let g = 50; g <= 250; g += 50) {{
          const y = yFor(g);
          svg += `<line x1="${{padL}}" y1="${{y}}" x2="${{w - padR}}" y2="${{y}}" stroke="var(--border)" stroke-dasharray="3,3"/>`;
          svg += `<text x="${{padL - 6}}" y="${{y + 4}}" text-anchor="end" fill="var(--text-muted)" font-size="10.5" font-family="var(--mono)">${{g}}</text>`;
        }}
        const n = suites.length;
        const slotW = (w - padL - padR) / Math.max(n, 1);
        suites.forEach((s, i) => {{
          const cx = padL + slotW * (i + 0.5);
          const hiKey = s.runs.c16 ? "c16" : "c8";
          const midKey = s.runs.c16 ? "c8" : "c4";
          const hiVal = (s.runs[hiKey] || {{}}).output_tps || 0;
          const midVal = (s.runs[midKey] || {{}}).output_tps || 0;
          const c1Val = (s.runs.c1 || {{}}).output_tps || 0;
          const bw = Math.min(14, slotW * 0.24);

          if (hiVal) {{
            const y = yFor(hiVal);
            svg += `<rect x="${{cx - bw * 1.5}}" y="${{y}}" width="${{bw}}" height="${{h - padB - y}}" rx="2" fill="var(--accent)"><title>${{esc(s.release)}} ${{hiKey}}: ${{hiVal}} tok/s</title></rect>`;
          }}
          if (midVal) {{
            const y = yFor(midVal);
            svg += `<rect x="${{cx - bw * 0.4}}" y="${{y}}" width="${{bw}}" height="${{h - padB - y}}" rx="2" fill="var(--cyan)"><title>${{esc(s.release)}} ${{midKey}}: ${{midVal}} tok/s</title></rect>`;
          }}
          if (c1Val) {{
            const y = yFor(c1Val);
            svg += `<rect x="${{cx + bw * 0.7}}" y="${{y}}" width="${{bw}}" height="${{h - padB - y}}" rx="2" fill="var(--amber)"><title>${{esc(s.release)}} c1: ${{c1Val}} tok/s</title></rect>`;
          }}
          svg += `<text x="${{cx}}" y="${{h - 16}}" text-anchor="middle" fill="var(--text-muted)" font-size="10.5" font-family="var(--mono)">${{esc(s.release.replace("spark-ds41f-", ""))}}</text>`;
        }});
        return `<svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="230">${{svg}}</svg>`;
      }}

      function renderGridSvg(baseSuite, candSuite) {{
        if (!candSuite.grid || !candSuite.grid.shapes.length) return "<p class='desc'>No engine grid in suite.</p>";
        const shapes = candSuite.grid.shapes;
        const baseMap = {{}};
        if (baseSuite.grid) baseSuite.grid.shapes.forEach(r => {{ baseMap[r.label] = r.step_ms; }});
        const w = 520, rowH = 19, padL = 165, padR = 72, padT = 8;
        const h = padT + shapes.length * rowH + 10;
        const maxMs = Math.max(55, ...shapes.map(s => s.step_ms));

        let svg = "";
        shapes.forEach((s, idx) => {{
          const y = padT + idx * rowH;
          const bMs = baseMap[s.label];
          const cMs = s.step_ms;
          const cw = (cMs / maxMs) * (w - padL - padR);
          const bw = bMs ? (bMs / maxMs) * (w - padL - padR) : 0;
          const delta = bMs ? (((cMs - bMs) / bMs) * 100).toFixed(1) + "%" : "";
          const shortLbl = s.label.replace("decode ", "").replace(" rows ", " ");
          svg += `<text x="${{padL - 8}}" y="${{y + 13}}" text-anchor="end" fill="var(--text-muted)" font-size="10.5" font-family="var(--mono)">${{esc(shortLbl)}}</text>`;
          if (bw > 0) {{
            svg += `<rect x="${{padL}}" y="${{y + 3}}" width="${{bw.toFixed(1)}}" height="5" rx="2" fill="var(--cyan)" opacity="0.55"><title>Base: ${{bMs.toFixed(2)}} ms</title></rect>`;
          }}
          svg += `<rect x="${{padL}}" y="${{y + 9}}" width="${{cw.toFixed(1)}}" height="6" rx="2" fill="var(--accent)"><title>Candidate: ${{cMs.toFixed(2)}} ms</title></rect>`;
          svg += `<text x="${{w - 6}}" y="${{y + 13}}" text-anchor="end" fill="var(--text)" font-size="10.5" font-family="var(--mono)">${{cMs.toFixed(1)}}ms ${{delta}}</text>`;
        }});
        return `<div style="max-height:230px;overflow-y:auto"><svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}">${{svg}}</svg></div>`;
      }}

      function renderSpecStallBox(baseSuite, candSuite) {{
        const hiKey = candSuite.runs.c16 ? "c16" : "c8";
        const run = candSuite.runs[hiKey] || {{}};
        const surv = run.survival || [];
        const cov = run.coverage || [];
        let htmlOut = `<div style="font-size:12.5px;margin-bottom:10px"><strong>DSpark Draft Survival / Coverage (${{esc(candSuite.release)}} ${{hiKey}}):</strong></div>`;
        htmlOut += `<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:14px">`;
        for (let k = 0; k < 5; k++) {{
          const sVal = surv[k] !== undefined ? (surv[k] * 100).toFixed(1) + "%" : "—";
          const cVal = cov[k] !== undefined ? (cov[k] * 100).toFixed(1) + "%" : "—";
          htmlOut += `<div style="background:var(--bg-elev);border:1px solid var(--border);border-radius:6px;padding:6px 8px;text-align:center">
            <div style="font-size:10.5px;color:var(--text-muted);font-family:var(--mono)">Draft k=${{k}}</div>
            <div style="font-size:13.5px;font-weight:700;color:var(--accent);font-family:var(--mono)">${{sVal}}</div>
            <div style="font-size:10.5px;color:var(--text-muted);font-family:var(--mono)">cov ${{cVal}}</div>
          </div>`;
        }}
        htmlOut += `</div>`;

        const st3 = candSuite.stalls["stall-3"];
        const st11 = candSuite.stalls["stall-11"];
        const st15 = candSuite.stalls["stall-15"];
        if (st3 || st15) {{
          htmlOut += `<div style="font-size:12.5px;margin-bottom:6px"><strong>131K Prefill Yield &amp; Mixed-Traffic Stalls (${{esc(candSuite.release)}}):</strong></div>`;
          htmlOut += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">`;
          if (st3) {{
            htmlOut += `<div style="background:var(--bg-elev);border:1px solid var(--border);border-radius:6px;padding:8px">
              <div style="font-size:10.5px;color:var(--text-muted)">3 Streams + 131K</div>
              <div style="font-size:14px;font-weight:700;font-family:var(--mono);color:var(--accent)">TTFT ${{st3.probe_ttft_p50_s.toFixed(2)}}s</div>
              <div style="font-size:11px;color:var(--text-muted);font-family:var(--mono)">${{Math.round(st3.prefill_tps).toLocaleString()}} tok/s prefill</div>
            </div>`;
          }}
          if (st11) {{
            htmlOut += `<div style="background:var(--bg-elev);border:1px solid var(--border);border-radius:6px;padding:8px">
              <div style="font-size:10.5px;color:var(--text-muted)">11 Streams + 131K</div>
              <div style="font-size:14px;font-weight:700;font-family:var(--mono);color:var(--accent)">TTFT ${{st11.probe_ttft_p50_s.toFixed(2)}}s</div>
              <div style="font-size:11px;color:var(--text-muted);font-family:var(--mono)">${{Math.round(st11.prefill_tps).toLocaleString()}} tok/s prefill</div>
            </div>`;
          }}
          if (st15) {{
            htmlOut += `<div style="background:var(--bg-elev);border:1px solid var(--border);border-radius:6px;padding:8px">
              <div style="font-size:10.5px;color:var(--text-muted)">15 Streams + 131K</div>
              <div style="font-size:14px;font-weight:700;font-family:var(--mono);color:var(--cyan)">${{Math.round(st15.prefill_tps).toLocaleString()}} tok/s</div>
              <div style="font-size:11px;color:var(--text-muted);font-family:var(--mono)">step p90 ${{st15.during_p90_s.toFixed(2)}}s</div>
            </div>`;
          }}
          htmlOut += `</div>`;
        }} else {{
          htmlOut += `<div style="font-size:12px;color:var(--text-muted)">Select <code>spark-ds41f-1030</code> on TP4 to inspect <code>stall-3</code>, <code>stall-11</code>, and <code>stall-15</code> prefill-yield probes.</div>`;
        }}
        return htmlOut;
      }}

      function renderChartsAndTable() {{
        const topo = topoSelect.value;
        const suites = DATA.suites.filter(s => s.topology === topo);
        const baseSuite = suites.find(s => s.id === baseSelect.value) || suites[0];
        const candSuite = suites.find(s => s.id === candSelect.value) || suites[suites.length - 1];

        document.getElementById("chart-prefill").innerHTML = renderPrefillSvg(suites, baseSuite, candSuite);
        document.getElementById("chart-decode").innerHTML = renderDecodeSvg(suites, baseSuite, candSuite);
        document.getElementById("chart-grid").innerHTML = renderGridSvg(baseSuite, candSuite);
        document.getElementById("chart-spec-stall").innerHTML = renderSpecStallBox(baseSuite, candSuite);

        document.getElementById("th-base").textContent = `${{baseSuite.release}} (${{baseSuite.topology}})`;
        document.getElementById("th-cand").textContent = `${{candSuite.release}} (${{candSuite.topology}})`;

        const rows = [];
        if (baseSuite.grid && candSuite.grid) {{
          rows.push({{
            name: `Decode step time geometric mean (${{candSuite.grid.shape_count}} shapes)`,
            b: `${{baseSuite.grid.geomean_ms.toFixed(2)}} ms`,
            c: `${{candSuite.grid.geomean_ms.toFixed(2)}} ms`,
            d: pctDelta(baseSuite.grid.geomean_ms, candSuite.grid.geomean_ms, true),
            note: "Dev-mode collective_rpc fixed-shape verification grid"
          }});
        }}
        [["c16", "Decode, 16 streams (output tok/s)"],
         ["c8", "Decode, 8 streams (output tok/s)"],
         ["c4", "Decode, 4 streams (output tok/s)"],
         ["c1", "Single-stream c1 (output tok/s)"],
         ["poisson", "Poisson arrivals (output tok/s)"]].forEach(([k, label]) => {{
          const br = baseSuite.runs[k], cr = candSuite.runs[k];
          if (br || cr) {{
            rows.push({{
              name: label,
              b: br ? `${{br.output_tps.toFixed(1)}}` : "—",
              c: cr ? `${{cr.output_tps.toFixed(1)}}` : "—",
              d: (br && cr) ? pctDelta(br.output_tps, cr.output_tps, false) : "—",
              note: cr && cr.ttft_p50_s ? `TTFT p50 ${{cr.ttft_p50_s.toFixed(3)}}s · TPOT p50 ${{cr.tpot_p50_ms.toFixed(1)}}ms` : ""
            }});
          }}
        }});
        [["c16", "Sampled decode, 16 streams (temp=1.0 tok/s)"],
         ["c8", "Sampled decode, 8 streams (temp=1.0 tok/s)"],
         ["c1", "Sampled decode, 1 stream (temp=1.0 tok/s)"]].forEach(([k, label]) => {{
          const br = (baseSuite.sampled || {{}})[k], cr = (candSuite.sampled || {{}})[k];
          if (br || cr) {{
            rows.push({{
              name: label,
              b: br ? `${{br.output_tps.toFixed(1)}}` : "—",
              c: cr ? `${{cr.output_tps.toFixed(1)}}` : "—",
              d: (br && cr) ? pctDelta(br.output_tps, cr.output_tps, false) : "—",
              note: "Seeded temperature 1.0, top_p 1.0"
            }});
          }}
        }});
        [32768, 131072, 262144, 524288, 500000, 1000000].forEach(sz => {{
          const bp = baseSuite.prefill.find(p => p.size === sz);
          const cp = candSuite.prefill.find(p => p.size === sz);
          if (bp || cp) {{
            const lbl = sz === 1000000 ? "1M" : `${{Math.round(sz / 1024)}}K`;
            rows.push({{
              name: `Source-text prefill, ${{lbl}} (tok/s)`,
              b: bp ? Math.round(bp.tps).toLocaleString() : "—",
              c: cp ? Math.round(cp.tps).toLocaleString() : "—",
              d: (bp && cp) ? pctDelta(bp.tps, cp.tps, false) : "—",
              note: "3 uncached Python stdlib prompts"
            }});
          }}
        }});

        document.getElementById("suite-compare-tbody").innerHTML = rows.map(r => `
          <tr>
            <td><strong>${{esc(r.name)}}</strong></td>
            <td class="num">${{esc(r.b)}}</td>
            <td class="num">${{esc(r.c)}}</td>
            <td class="num">${{r.d || "—"}}</td>
            <td style="color:var(--text-muted);font-size:12px">${{esc(r.note)}}</td>
          </tr>
        `).join("");
      }}

      topoSelect.addEventListener("change", populateSuiteSelectors);
      baseSelect.addEventListener("change", renderChartsAndTable);
      candSelect.addEventListener("change", renderChartsAndTable);
      populateSuiteSelectors();

      // Render Shapes & KV Capacity
      const maxBlocks = Math.max(...DATA.shapes.map(s => s.kv_capacity.pool_blocks));
      document.getElementById("shape-cards").innerHTML = DATA.shapes.map(s => {{
        const kv = s.kv_capacity;
        const pct = Math.round((kv.pool_blocks / maxBlocks) * 100);
        return `
          <div class="shape-card ${{s.default ? 'default' : ''}}">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <strong style="font-size:16px;font-family:var(--mono)">${{esc(s.name)}}</strong>
              ${{s.default ? '<span class="badge live">Default Shape</span>' : `<span class="badge">${{esc(s.page_size_label)}} pages</span>`}}
            </div>
            <div style="font-size:12.5px;color:var(--text-muted);margin-top:4px">
              ${{s.nodes}}× DGX Spark &bull; Transport <code>${{esc(s.transport)}}</code> &bull; Kernel <code>${{esc(s.kernel_release)}}</code>
            </div>
            <div class="kv-bar"><div class="kv-bar-fill" style="width:${{pct}}%"></div></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12.5px">
              <div>KV Pool / Rank: <strong>${{esc(s.kv_cache_gib)}}</strong> (${{kv.pool_blocks.toLocaleString()}} blocks)</div>
              <div>Context Limit: <strong>${{s.max_model_len.toLocaleString()}}</strong> tok</div>
              <div>Full Contexts Held: <strong>${{kv.full_contexts}} (${{kv.contexts.toFixed(2)}}×)</strong></div>
              <div>All ${{s.max_num_seqs}} Seqs Fit: <strong>${{kv.every_sequence_tokens.toLocaleString()}}</strong> tok</div>
              <div>Batch Budget: <strong>${{s.max_num_batched_tokens.toLocaleString()}}</strong> tok</div>
              <div>CUDA Graphs: <strong>1–${{s.max_cudagraph_capture_size}}</strong> rows</div>
            </div>
          </div>
        `;
      }}).join("");

      document.getElementById("shapes-tbody").innerHTML = DATA.shapes.map(s => {{
        const kv = s.kv_capacity;
        return `
          <tr>
            <td><code>${{esc(s.name)}}</code>${{s.default ? ' <span class="badge live">default</span>' : ''}}</td>
            <td>TP${{s.nodes}} <code>${{esc(s.transport)}}</code></td>
            <td><code>${{esc(s.kernel_release)}}</code> (${{esc(s.page_size_label)}})</td>
            <td class="num">${{s.max_model_len.toLocaleString()}}</td>
            <td class="num">${{s.max_num_seqs}}</td>
            <td class="num">${{s.max_num_batched_tokens.toLocaleString()}}</td>
            <td class="num">${{esc(s.kv_cache_gib)}}</td>
            <td class="num">${{kv.pool_blocks.toLocaleString()}}</td>
            <td class="num"><strong>${{kv.full_contexts}}</strong> (${{kv.contexts.toFixed(2)}}×)</td>
            <td class="num">${{kv.vllm_capacity_tokens.toLocaleString()}} (${{kv.vllm_capacity_contexts.toFixed(2)}}×)</td>
            <td>${{s.qualified_release ? `<code>${{esc(s.qualified_release)}}</code>` : '<span style="color:var(--text-muted)">unqualified</span>'}}</td>
          </tr>
        `;
      }}).join("");

      // Render Upstreams & Runtime Choices
      document.getElementById("upstreams-tbody").innerHTML = DATA.upstreams.map(u => `
        <tr>
          <td><strong><code>${{esc(u.name)}}</code></strong></td>
          <td>${{esc(u.version || u.kind)}}</td>
          <td class="num"><code>${{esc((u.revision || "—").slice(0, 12))}}</code></td>
          <td class="num">${{u.patch_count !== null ? `<strong>${{u.patch_count}}</strong> patches` : "—"}}</td>
          <td class="num">${{u.tree ? `<code>${{esc(u.tree.slice(0, 10))}}</code>` : "—"}}</td>
          <td style="font-size:12px;color:var(--text-muted)">${{esc(u.role)}}</td>
        </tr>
      `).join("");

      const choiceEntries = Object.entries(DATA.runtime_choices);
      function renderChoices(filterText) {{
        const q = (filterText || "").toLowerCase();
        const filtered = choiceEntries.filter(([k, v]) =>
          k.toLowerCase().includes(q) || JSON.stringify(v).toLowerCase().includes(q)
        );
        document.getElementById("choices-tbody").innerHTML = filtered.map(([k, v]) => `
          <tr>
            <td><code>${{esc(k)}}</code></td>
            <td style="font-size:12.5px">${{esc(typeof v === "object" ? JSON.stringify(v) : v)}}</td>
          </tr>
        `).join("");

        const caps = DATA.capabilities.filter(c => c.toLowerCase().includes(q));
        document.getElementById("cap-tags").innerHTML = caps.map(c =>
          `<span class="cap-tag">${{esc(c)}}</span>`
        ).join("");
      }}
      renderChoices("");
      document.getElementById("choice-search").addEventListener("input", e => renderChoices(e.target.value));

      // Render Timeline
      function renderTimeline(filterText) {{
        const q = (filterText || "").toLowerCase();
        const items = DATA.timeline.filter(t =>
          t.stem.toLowerCase().includes(q) ||
          (t.change_from_previous || "").toLowerCase().includes(q) ||
          (t.evidence_summary || "").toLowerCase().includes(q)
        );
        document.getElementById("timeline-list").innerHTML = items.map(t => `
          <div class="timeline-item ${{t.is_promoted ? 'promoted' : ''}}">
            <div class="timeline-head">
              <div>
                <strong style="font-size:15px;font-family:var(--mono)">${{esc(t.short_name)}}</strong>
                <span style="color:var(--text-muted);font-size:12.5px;margin-left:8px">${{esc(t.stem)}}</span>
                ${{t.is_promoted ? '<span class="badge live" style="margin-left:8px">Active Production</span>' : ''}}
              </div>
              <div style="font-family:var(--mono);font-size:12px;color:var(--text-muted)">
                Shape: <code>${{esc(t.shape)}}</code> &bull; Image: <code>${{esc(t.image_tag || "—")}}</code> ${{t.vllm_tree ? `&bull; vLLM tree <code>${{esc(t.vllm_tree)}}</code>` : ''}}
              </div>
            </div>
            ${{t.change_from_previous ? `<div style="font-size:13px;margin-bottom:6px"><strong>Changes:</strong> ${{esc(t.change_from_previous)}}</div>` : ''}}
            ${{t.evidence_summary ? `<div style="font-size:12.5px;color:var(--text-muted)"><strong>Qualification:</strong> ${{esc(t.evidence_summary)}}</div>` : ''}}
          </div>
        `).join("");
      }}
      renderTimeline("");
      document.getElementById("timeline-search").addEventListener("input", e => renderTimeline(e.target.value));
    }})();
  </script>
</body>
</html>
"""


def render_targets(data: dict[str, Any]) -> dict[Path, str]:
    readme_path = ROOT / "README.md"
    current_state_path = ROOT / "docs/current-state.md"
    todo_path = ROOT / "TODO.md"
    site_index_path = ROOT / "docs/index.html"
    nojekyll_path = ROOT / "docs/.nojekyll"

    readme_text = _replace_markers(
        readme_path.read_text(),
        {
            "current-baseline-summary": render_readme_current_baseline(data),
            "performance-summary": render_readme_performance_summary(data),
        },
        "README.md",
    )
    current_state_text = _replace_markers(
        current_state_path.read_text(),
        {
            "active-settings": render_current_state_active_settings(data),
        },
        "docs/current-state.md",
    )
    todo_text = _replace_markers(
        todo_path.read_text(),
        {
            "next-release-status": render_todo_next_release_status(data),
            "current-reference": render_todo_current_reference(data),
        },
        "TODO.md",
    )
    site_html = render_site_html(data)

    return {
        readme_path: readme_text,
        current_state_path: current_state_text,
        todo_path: todo_text,
        site_index_path: site_html,
        nojekyll_path: "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that generated blocks and docs/index.html match the manifests without writing",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        help="also write index.html, data.json, and .nojekyll into this directory for GitHub Pages deployment",
    )
    args = parser.parse_args(argv)

    data = collect_manifest_data()
    targets = render_targets(data)

    if args.check:
        drifted: list[str] = []
        for path, expected in targets.items():
            rel = path.relative_to(ROOT)
            if not path.exists():
                drifted.append(f"{rel}: file is missing (run python3 scripts/render_docs.py)")
                continue
            actual = path.read_text()
            if actual != expected:
                diff = "\n".join(
                    difflib.unified_diff(
                        actual.splitlines(),
                        expected.splitlines(),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                        n=3,
                    )
                )
                drifted.append(f"{rel} is out of date with manifests:\n{diff[:2000]}")
        if drifted:
            for msg in drifted:
                print(msg, file=sys.stderr)
            return 1
        return 0

    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    if args.site_dir:
        out_dir = args.site_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(targets[ROOT / "docs/index.html"])
        (out_dir / ".nojekyll").write_text("")
        payload = {
            k: v for k, v in data.items() if k != "suites_by_path"
        }
        (out_dir / "data.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
