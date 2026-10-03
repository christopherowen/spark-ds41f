"""Reproduce the comparison from native reports; reject mismatched workloads.

Usage: python3 summarize.py RAW_DIRECTORY OUTPUT_JSON
RAW_DIRECTORY contains bench-tp4.json, prefill-cooled.json, prefill-only.json
and prefix-only.json. Both thermal aborts remain in the run inventory.
The immutable TP3 reference remains in manifests/benchmarks/.
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json"


def main():
    raw, output = map(Path, sys.argv[1:])
    cli = runpy.run_path(str(ROOT / "bin/spark3"))
    reference = json.loads(REFERENCE.read_text())
    measured = json.loads((raw / "bench-tp4.json").read_text())
    combined = json.loads((raw / "prefill-cooled.json").read_text())
    cooled = json.loads((raw / "prefill-only.json").read_text())
    prefix = json.loads((raw / "prefix-only.json").read_text())
    for report in (cooled, prefix):
        assert not report.get("aborted")
        assert not any(t["thermal_slowdown_ms"] for t in report["thermal"].values())
    checks = {}
    for name, report in (("main", measured), ("prefill", cooled), ("prefix", prefix)):
        assert report["identity"]["client_host"] == reference["identity"]["client_host"] == "dgx1"
        assert not report["identity"]["live_problems"]
        for field in ("seed", "min_samples", "max_samples", "concurrency", "decode_cases",
                      "prefill_text", "prefill_sizes", "prefill_repeats", "prefix_tokens",
                      "prompts_sha256"):
            assert report["workload"][field] == reference["workload"][field], (name, field)
        checks[name + "_workload_matches"] = True

    def comparison(current, previous):
        return {
            "tp3": previous,
            "tp4": current,
            "change_pct_and_welch95": [round(v, 3) for v in cli["difference"](current, previous)],
        }

    result = {
        "control": "historical TP3 triangle, 2026-10-02; not a fresh TP-only A/B",
        "reference": str(REFERENCE.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "decode": {}, "prefill": {}, "prefix": {}, "checks": checks,
        "runs": {
            name: {k: report.get(k) for k in ("started_utc", "finished_utc", "identity",
                                            "cooling", "aborted", "thermal", "memory",
                                            "discarded_samples")}
            for name, report in (("main", measured), ("combined_cooled", combined),
                                 ("prefill_only", cooled), ("prefix_only", prefix))
        },
    }
    for name, point in measured["suites"]["decode"]["points"].items():
        old = reference["suites"]["decode"]["points"][name]
        new_samples = measured["suites"]["decode"]["samples"][name]
        old_samples = reference["suites"]["decode"]["samples"][name]
        def lengths(samples):
            return [[(r["prompt_tokens"], r["completion_tokens"], r["ok"]) for r in s["requests"]]
                    for s in samples]
        assert lengths(new_samples) == lengths(old_samples), name
        result["decode"][name] = {
            metric: comparison(point[metric], old[metric])
            for metric in ("tps", "decode_window_tps", "ttft_s", "accepted_per_draft",
                           "verified_per_draft", "step_ms") if point[metric]["n"]
        }
        result["decode"][name]["reasoning_share"] = [old["reasoning_share"], point["reasoning_share"]]
    checks["decode_request_lengths_match"] = True

    for size, point in cooled["suites"].get("prefill", {}).get("points", {}).items():
        old = reference["suites"]["prefill"]["points"][size]
        actual = [s["prompt_tokens"] for s in cooled["suites"]["prefill"]["samples"][size]]
        expected = [s["prompt_tokens"] for s in reference["suites"]["prefill"]["samples"][size]]
        assert actual == expected, (size, actual, expected)
        assert point["failed_requests"] == 0
        result["prefill"][size] = {metric: comparison(point[metric], old[metric])
                                   for metric in ("prefill_tps", "ttft_s")}
        result["prefill"][size]["actual_prompt_tokens"] = actual
    checks["prefill_request_lengths_match"] = bool(result["prefill"])
    if "prefix" in prefix["suites"]:
        for metric in ("cold_ttft_s", "warm_ttft_s", "warm_hit_rate"):
            result["prefix"][metric] = comparison(prefix["suites"]["prefix"][metric],
                                                   reference["suites"]["prefix"][metric])
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
