#!/usr/bin/env python3
"""Offline assessment of the preserved replay exports; never contacts a server."""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import agentx  # noqa: E402

RUN = HERE / "runs/2026-11-04-spaceport/attempt-2"


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def analyze(label):
    point = RUN / "points" / label
    data = rows(point / "harness/profile_export.jsonl.gz")
    series = rows(point / "server-series.jsonl.gz")
    successful = [r for r in data if not r.get("error")]
    profiled = [r for r in successful if r["metadata"]["benchmark_phase"] == "profiling"]
    span = agentx.profiling_window(profiled)
    samples = agentx.window(series, *span) if span else []
    repetition = 'vllm:request_success_total{finished_reason="repetition"}'
    increases = [(a, b) for a, b in zip(series, series[1:])
                 if b["m"].get(repetition, 0) > a["m"].get(repetition, 0)]
    mismatches = []
    for r in profiled:
        diff = agentx.value(r, "osl_mismatch_diff_pct") or 0
        if diff == 0:
            continue
        end = r["metadata"]["request_end_ns"] / 1e9
        matches = [(a, b) for a, b in increases if a["t"] < end <= b["t"]]
        actual = agentx.value(r, "output_sequence_length")
        mismatches.append({
            "request_id": r["metadata"]["x_request_id"],
            "source_trace_id": r["metadata"]["source_trace_id"],
            "ended_utc": utc(end), "actual_tokens": actual,
            "target_tokens_derived_from_diff": round(actual / (1 + diff / 100)),
            "osl_diff_pct": diff, "was_cancelled": r["metadata"]["was_cancelled"],
            "repetition_counter_intervals": [
                {"start_utc": utc(a["t"]), "end_utc": utc(b["t"]),
                 "increments": b["m"][repetition] - a["m"].get(repetition, 0)}
                for a, b in matches],
        })
    errors = [{"metadata": r["metadata"], "error": r["error"],
               "request_end_utc": utc(r["metadata"]["request_end_ns"] / 1e9)}
              for r in data if r.get("error")]
    reason_keys = [k for k in series[0]["m"] if k.startswith("vllm:request_success_total{")]
    telemetry = load(point / "point.telemetry.json")
    return {
        "exported_records": len(data), "successful_records": len(successful),
        "profiling_records": len(profiled), "errors": errors,
        "exported_cancelled_records": sum(r["metadata"]["was_cancelled"] for r in data),
        "server_profiling": agentx.server_summary(samples),
        "server_whole_run": agentx.server_summary(series),
        "finish_reasons_profiling": {k: agentx.delta(samples, k) for k in reason_keys} if samples else {},
        "finish_reasons_whole_run": {k: agentx.delta(series, k) for k in reason_keys},
        "miss_cost": agentx.miss_cost(profiled), "output_mismatches": mismatches,
        "conditions": load(point / "conditions.json"),
        "client_memory_peak_gib": load(point / "run.json")["client_memory_peak_bytes"] / 2**30,
        "auxiliary_telemetry": {
            "duration_s": telemetry["seconds"],
            "nodes": {node: {"samples": d["samples"],
                              "max_gpu_c": d["temperature.gpu"]["max"],
                              "thread_window_s": d["threads"]["window_s"]}
                      for node, d in telemetry["nodes"].items()},
            "limitation": "Uneven c8 GPU coverage and short thread windows; cannot establish full-run CPU/clock bottlenecks. Continuous guard conditions are separate.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare recomputation with committed assessment")
    args = parser.parse_args()
    assessment = {label: analyze(label) for label in ["c38-boot-a-c8", "c38-boot-a-c16"]}
    c8, c16 = assessment.values()
    # Evidence consistency checks, including failures and incomplete points.
    assert c8["profiling_records"] == 765 and not c8["errors"]
    assert len(c8["output_mismatches"]) == 7
    assert all(len(r["repetition_counter_intervals"]) == 1 and
               r["repetition_counter_intervals"][0]["increments"] == 1 for r in c8["output_mismatches"])
    assert not c16["profiling_records"] and len(c16["errors"]) == 1
    assert c16["errors"][0]["request_end_utc"].startswith("2026-11-04T02:16:01")
    assert c16["conditions"]["halted"] == ["sp2: GPU at 78 C is within 5 C of its thermal slowdown limit"]
    restoration = load(RUN / "restoration.json")
    assert restoration["all_ranks_restored"] and restoration["live_doctor"]["exit"] == 0
    assert len(restoration["nodes"]) == 4 and not restoration["holds_present"]
    assert all(n["running"] and n["clean"] and not n["oom_killed"] for n in restoration["nodes"])
    receipt = load(RUN / "window/receipt.json")
    assert receipt["restored"] and len(receipt["points"]) == 1 and receipt["error"]
    assert receipt["points"][0]["result"]["harness"]["submission_valid"]
    out = RUN / "assessment.json"
    if args.check:
        assert load(out) == assessment, "saved assessment differs from preserved exports"
    else:
        out.write_text(json.dumps(assessment, indent=2) + "\n")
    print("Offline exports agree: c8 valid, seven matched repetition stops; c16 incomplete with one prior no-content error; four ranks restored.")


if __name__ == "__main__":
    main()
