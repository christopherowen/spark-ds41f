"""Summarize fresh TP3 receipts against the retained TP3 and TP4 measurements.

Usage: TMPDIR=/tmp python3 summarize.py RAW_DIRECTORY OUTPUT_JSON
RAW_DIRECTORY contains decode.json, prefill.json and prefix.json.
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    raw, output = map(Path, sys.argv[1:])
    cli = runpy.run_path(str(ROOT / "bin/spark3"))
    baseline = ROOT / "manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json"
    old = json.loads(baseline.read_text())
    tp4_path = ROOT / "experiments/2026-10-03-tp3-tp4-comparison/results.json"
    tp4 = json.loads(tp4_path.read_text())
    reports = {name: json.loads((raw / f"{name}.json").read_text())
               for name in ("decode", "prefill", "prefix")}
    result = {
        "scope": "Fresh TP3 triangle screen; TP4 is the earlier retained run, not a simultaneous A/B.",
        "references_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (baseline, tp4_path)},
        "runs": {}, "decode": {}, "prefill": {}, "prefix": {},
    }

    def compare(now, prior):
        return {"fresh_tp3": now, "previous": prior,
                "fresh_tp3_change_pct_and_welch95": list(cli["difference"](now, prior))}

    for name, report in reports.items():
        assert report["identity"]["client_host"] == "dgx1"
        assert not report["identity"]["live_problems"]
        # A failed long-prefill screen is evidence, not a reason to erase the
        # successful decode/prefix results or retry until a report passes.
        if name != "prefill":
            assert not report.get("aborted"), (name, report.get("aborted"))
        assert not any(t["thermal_slowdown_ms"] for t in report["thermal"].values()), name
        for field in ("seed", "min_samples", "max_samples", "concurrency", "decode_cases",
                      "prompts_sha256"):
            assert report["workload"][field] == old["workload"][field], (name, field)
        result["runs"][name] = {k: report.get(k) for k in (
            "started_utc", "finished_utc", "identity", "cooling", "aborted", "thermal",
            "memory", "discarded_samples")}
    assert reports["decode"]["suites"]["quality"]["ok"]
    for point, current in reports["decode"]["suites"]["decode"]["points"].items():
        assert current["failed_requests"] == 0
        samples = reports["decode"]["suites"]["decode"]["samples"][point]
        prior_samples = old["suites"]["decode"]["samples"][point]
        def lengths(rows):
            return [[(r["prompt_tokens"], r["completion_tokens"], r["ok"])
                     for r in s["requests"]] for s in rows]
        assert lengths(samples) == lengths(prior_samples), point
        result["decode"][point] = {}
        for metric in ("tps", "decode_window_tps", "ttft_s", "step_ms", "accepted_per_draft",
                       "verified_per_draft"):
            if not current[metric]["n"]:
                continue
            result["decode"][point][metric] = {
                "versus_previous_tp3": compare(current[metric], old["suites"]["decode"]["points"][point][metric]),
                "versus_previous_tp4": compare(current[metric], tp4["decode"][point][metric]["tp4"]),
            }
    result["prefill_completed"] = not reports["prefill"].get("aborted")
    for size, current in reports["prefill"]["suites"].get("prefill", {}).get("points", {}).items():
        assert current["failed_requests"] == 0
        lengths = [r["prompt_tokens"] for r in reports["prefill"]["suites"]["prefill"]["samples"][size]]
        assert lengths == tp4["prefill"][size]["actual_prompt_tokens"], (size, lengths)
        result["prefill"][size] = {"actual_prompt_tokens": lengths}
        for metric in ("prefill_tps", "ttft_s"):
            result["prefill"][size][metric] = {
                "versus_previous_tp3": compare(current[metric], old["suites"]["prefill"]["points"][size][metric]),
                "versus_previous_tp4": compare(current[metric], tp4["prefill"][size][metric]["tp4"]),
            }
    for metric in ("cold_ttft_s", "warm_ttft_s", "warm_hit_rate"):
        current = reports["prefix"]["suites"]["prefix"][metric]
        result["prefix"][metric] = {
            "versus_previous_tp3": compare(current, old["suites"]["prefix"][metric]),
            "versus_previous_tp4": compare(current, tp4["prefix"][metric]["tp4"]),
        }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
