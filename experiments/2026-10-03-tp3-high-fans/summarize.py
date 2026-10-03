"""Reproduce the high-fan comparison without dropping incomplete suites.

Usage: TMPDIR=/tmp python3 summarize.py RAW_DIRECTORY OUTPUT_JSON
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    raw, destination = map(Path, sys.argv[1:])
    cli = runpy.run_path(str(ROOT / "bin/spark3"))
    report = json.loads((raw / "bench.json").read_text())
    main_path = ROOT / "manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json"
    previous_main = json.loads(main_path.read_text())
    archive = ROOT / "experiments/2026-10-03-tp3-revalidation/hardware/runs.tar.gz"
    with tarfile.open(archive) as bundle:
        previous = {name: json.load(bundle.extractfile(name + ".json"))
                    for name in ("decode", "prefill", "prefix")}
    elapsed = json.loads((raw / "cooldown-complete.json").read_text())["elapsed_seconds"]
    assert elapsed >= 1200
    assert report["identity"]["client_commit"] == previous["decode"]["identity"]["client_commit"]
    assert report["identity"]["cluster_config_sha256"] == previous["decode"]["identity"]["cluster_config_sha256"]
    expected_fan_errors = {host + ': dgx-fan-control.service is enabled/inactive, expected enabled/active; run sudo systemctl enable --now dgx-fan-control.service'
                           for host in ("dgx1", "dgx2", "dgx3")}
    assert set(report["identity"]["live_problems"]) == expected_fan_errors
    for field in ("seed", "min_samples", "max_samples", "concurrency", "decode_cases",
                  "prefill_text", "prefill_sizes", "prefill_repeats", "prefix_tokens", "prompts_sha256"):
        assert report["workload"][field] == previous_main["workload"][field], field
    result = {
        "scope": "Same TP3/r5o serving recipe; 20-minute idle maximum-fan cooldown and maximum fans throughout the combined benchmark.",
        "limits": "One boot per arm, three decode samples; speculative costs reprofiled on startup. Separately cooled previous suites versus one continuous high-fan suite.",
        "reference_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (main_path, archive)},
        "cooldown_seconds": elapsed,
        "run": {k: report.get(k) for k in ("started_utc", "finished_utc", "identity", "aborted",
                                           "thermal", "memory", "discarded_samples")},
        "completed_suites": list(report["suites"]),
        "quality": report["suites"].get("quality"),
        "decode": {}, "prefill": {}, "prefix": {},
    }
    fan_samples = [json.loads(line) for line in (raw / "fans.jsonl").read_text().splitlines()]
    result["fan_observations"] = {"sample_count": len(fan_samples), "nodes": {}}
    for host in ("dgx1", "dgx2", "dgx3"):
        readings = [node["output"].splitlines() for sample in fan_samples
                    for node in sample["nodes"] if node["node"] == host]
        result["fan_observations"]["nodes"][host] = {
            "maximum_state_every_sample": all("state=12/12" in lines[0] for lines in readings),
            "minimum_recorded_rpm": [min(int(lines[i]) for lines in readings) for i in (2, 3)],
        }

    def compare(current, baseline):
        return {"high_fans": current, "reference": baseline,
                "change_pct_and_welch95": list(cli["difference"](current, baseline))}

    for name, point in report["suites"].get("decode", {}).get("points", {}).items():
        result["decode"][name] = {"failed_requests": point["failed_requests"],
                                  "reasoning_share": point["reasoning_share"]}
        for metric in ("tps", "decode_window_tps", "ttft_s", "step_ms", "accepted_per_draft", "verified_per_draft"):
            if point[metric]["n"]:
                result["decode"][name][metric] = {
                    "versus_main": compare(point[metric], previous_main["suites"]["decode"]["points"][name][metric]),
                    "versus_ordinary_fans": compare(point[metric], previous["decode"]["suites"]["decode"]["points"][name][metric]),
                }
    for size, point in report["suites"].get("prefill", {}).get("points", {}).items():
        lengths = [r["prompt_tokens"] for r in report["suites"]["prefill"]["samples"][size]]
        assert lengths == [r["prompt_tokens"] for r in previous_main["suites"]["prefill"]["samples"][size]], size
        result["prefill"][size] = {"actual_prompt_tokens": lengths, "failed_requests": point["failed_requests"]}
        for metric in ("prefill_tps", "ttft_s"):
            result["prefill"][size][metric] = compare(point[metric], previous_main["suites"]["prefill"]["points"][size][metric])
    if "prefix" in report["suites"]:
        for metric in ("cold_ttft_s", "warm_ttft_s", "warm_hit_rate"):
            result["prefix"][metric] = {
                "versus_main": compare(report["suites"]["prefix"][metric], previous_main["suites"]["prefix"][metric]),
                "versus_ordinary_fans": compare(report["suites"]["prefix"][metric], previous["prefix"]["suites"]["prefix"][metric]),
            }
    destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
