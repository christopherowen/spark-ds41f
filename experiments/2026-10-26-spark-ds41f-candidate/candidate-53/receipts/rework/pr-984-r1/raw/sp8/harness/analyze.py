#!/usr/bin/env python3
"""Summarize boot_tp3r0.py runs: phase wall times, the UMA-release and trim calls by
phase, memory at readiness, and per-arm means with a two-boot interval."""

import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
runs = sys.argv[2:]  # label=arm


def load(label):
    return json.loads((out / f"{label}.json").read_text())


def phase_seconds(d, name, nth=None):
    ps = [p for p in d["phases"] if p["name"] == name]
    if nth is not None:
        return ps[nth]["seconds"] if len(ps) > nth else None
    return round(sum(p["seconds"] for p in ps), 3) if ps else None


def ops(d, op, phase=None, key="seconds"):
    return round(sum(o[key] or 0 for o in d["ops_by_phase"]
                     if o["op"] == op and (phase is None or o["phase"] == phase)), 4)


rows = {}
for spec in runs:
    label, arm = spec.split("=")
    d = load(label)
    st = d["proc_status_kib_at_ready"]
    cu = d["cuda_at_ready"]
    rows[label] = {
        "arm": arm,
        "error": d["error"],
        "total_to_ready_s": d["seconds"]["total_to_ready"],
        "init_device_s": phase_seconds(d, "init_device"),
        "load_model_s": phase_seconds(d, "load_model"),
        "post_process_target_s": phase_seconds(d, "process_weights_after_loading", 0),
        "post_process_draft_s": phase_seconds(d, "process_weights_after_loading", 1),
        "determine_available_memory_s": phase_seconds(d, "determine_available_memory"),
        "prepare_kernels_s": phase_seconds(d, "_prepare_kernels"),
        "initialize_from_config_s": phase_seconds(d, "initialize_from_config"),
        "compile_or_warm_up_s": phase_seconds(d, "compile_or_warm_up_model"),
        "capture_model_s": phase_seconds(d, "runner.capture_model"),
        "warmup_kernels_s": phase_seconds(d, "warmup_kernels"),
        "freeze_gc_heap_s": phase_seconds(d, "freeze_gc_heap"),
        "uma_release_calls_post_process": ops(d, "release_device_memory_under_pressure",
                                              "process_weights_after_loading", "n"),
        "uma_release_fired_post_process": ops(d, "release_device_memory_under_pressure",
                                              "process_weights_after_loading", "fired"),
        "uma_release_s_post_process": ops(d, "release_device_memory_under_pressure",
                                          "process_weights_after_loading"),
        "gc_collect_n": ops(d, "gc.collect", None, "n"),
        "gc_collect_s": ops(d, "gc.collect"),
        "gc_collect_s_post_process": ops(d, "gc.collect", "process_weights_after_loading"),
        "trim_host_heap_n": ops(d, "trim_host_heap", None, "n"),
        "trim_host_heap_s": ops(d, "trim_host_heap"),
        "mi_collect_s": ops(d, "mi_collect"),
        "malloc_trim_s": ops(d, "malloc_trim"),
        "trim_rss_freed_mib_total": ops(d, "trim_host_heap", None, "rss_freed_mib"),
        "trim_rss_freed_mib_freeze_gc_heap": ops(d, "trim_host_heap", "freeze_gc_heap",
                                                 "rss_freed_mib"),
        "mi_collect_rss_freed_mib_freeze_gc_heap": ops(d, "mi_collect", "freeze_gc_heap",
                                                       "rss_freed_mib"),
        "malloc_trim_rss_freed_mib_freeze_gc_heap": ops(d, "malloc_trim", "freeze_gc_heap",
                                                        "rss_freed_mib"),
        "memavail_min_startup_gib": d["memavailable"]["min_available_gib_startup"],
        "memavail_min_at_s": d["memavailable"]["min_available_at_s"],
        "memavail_at_ready_gib": d["memavailable"]["available_at_ready_gib"],
        "memavail_settled_gib": d["memavailable"]["available_settled_gib"],
        "memavail_first_gib": d["memavailable"]["first_available_gib"],
        "vmrss_at_ready_mib": round(st.get("VmRSS", 0) / 1024, 1),
        "rss_anon_at_ready_mib": round(st.get("RssAnon", 0) / 1024, 1),
        "cuda_reserved_gib": round(cu.get("reserved_bytes", 0) / 2**30, 3),
        "cuda_slack_mib": round((cu.get("reserved_bytes", 0) - cu.get("allocated_bytes", 0))
                                / 2**20, 1),
    }

arms = {}
for label, r in rows.items():
    arms.setdefault(r["arm"], []).append(label)


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    half = (max(vals) - min(vals)) / 2 if len(vals) > 1 else None
    return {"mean": round(m, 3), "boots": vals, "half_range": None if half is None else round(half, 3)}


summary = {}
metrics = [k for k in next(iter(rows.values())) if k not in ("arm", "error")]
for arm, labels in arms.items():
    summary[arm] = {m: stat([rows[l][m] for l in labels]) for m in metrics}
print(json.dumps({"runs": rows, "by_arm": summary}, indent=1))
