#!/usr/bin/env python3
"""Summarize synthetic workloads, with intervals across IDs, not serving boots."""
import json
from pathlib import Path
import statistics


def interval(values):
    # Student t, two-sided 95%, 9 degrees of freedom; 10 non-warmup steps.
    assert len(values) == 10
    center = statistics.mean(values)
    half = 2.2621571628540993 * statistics.stdev(values) / len(values) ** 0.5
    return {"mean": center, "ci95": [center - half, center + half]}


root = Path(__file__).parent
output = {"scope": "CPU native-planner fixtures; no physical reads, GPU or serving boots",
          "interval_scope": "10 seeded ID batches after the first warmup; repeated processes use identical IDs and are not independent boot replicates",
          "runs": {}}
for run in ("pinned-a", "pinned-b"):
    receipt = json.loads((root / "runs" / run / "receipt.json").read_text())
    cases = []
    for pair in receipt["runs"]:
        b, c = pair["base"], pair["candidate"]
        before, after = b["samples"][1:], c["samples"][1:]
        case = {k: b[k] for k in ("tokens", "tp", "hot_pct", "populated")}
        case["base_planned_jobs"] = interval([x["planned_jobs"] for x in before])
        case["candidate_planned_jobs"] = interval([x["planned_jobs"] for x in after])
        case["paired_jobs_saved"] = interval([x["planned_jobs"] - y["planned_jobs"] for x, y in zip(before, after)])
        case["paired_planner_delta_us"] = interval([(y["plan_ns"] - x["plan_ns"]) / 1000 for x, y in zip(before, after)])
        case["base_planner_us"] = interval([x["plan_ns"] / 1000 for x in before])
        case["candidate_planner_us"] = interval([x["plan_ns"] / 1000 for x in after])
        case["planned_job_reduction_pct"] = 100 * (1 - case["candidate_planned_jobs"]["mean"] / case["base_planned_jobs"]["mean"]) if case["base_planned_jobs"]["mean"] else 0
        case["base_peak_rss_bytes"] = b["peak_rss_bytes"]
        case["candidate_peak_rss_bytes"] = c["peak_rss_bytes"]
        assert b["cache_allocation_bytes"] == c["cache_allocation_bytes"] == 17956864
        assert all(b[k] and c[k] for k in ("byte_checks", "cache_unchanged", "missing_source_rejected"))
        cases.append(case)
    output["runs"][run] = cases
(root / "results.json").write_text(json.dumps(output, indent=2) + "\n")
for run, cases in output["runs"].items():
    for c in cases:
        if c["tokens"] == 96:
            print(run, c["tp"], c["hot_pct"], "jobs", c["base_planned_jobs"], "->", c["candidate_planned_jobs"],
                  "saved", c["paired_jobs_saved"], "plan delta us", c["paired_planner_delta_us"])
