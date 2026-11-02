#!/usr/bin/env python3
"""Summarize runs/*.json of bench_context_ladder.py: per shape, the ladder arm's replay
time against the base arm's, paired within each process (both arms share it), with a
95% confidence interval over the processes; and each arm's capture cost when it was
captured first (a fresh graph pool)."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Two-sided 95% Student t quantiles by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def interval(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, float("nan")
    return mean, T95[len(values) - 1] * statistics.stdev(values) / len(values) ** 0.5


def main(directory):
    runs = defaultdict(list)
    memory = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        if "replay" in record:
            runs[record["shape"]].append((path.name, record))
        elif "memory" in record:
            memory[record["shape"]][record["memory"]["arm"]].append((path.name, record["memory"]))
    summary = {}
    for shape, arms in sorted(memory.items()):
        # Per copy: device memory beyond the allocator's (graph executables and other
        # driver allocations), and the allocator's (the aux buffer and graph pool).
        summary[f"{shape}-memory"] = {
            arm: {
                "runs": [name for name, _ in records],
                "copies": records[0][1]["copies"],
                "graphs_per_copy": records[0][1]["graphs_per_copy"],
                "aux_mib_per_copy": records[0][1]["aux_mib_per_copy"],
                "torch_reserved_mib_per_copy": [
                    round(m["torch_reserved_delta_mib"] / m["copies"], 3) for _, m in records
                ],
                "device_beyond_reserved_mib_per_copy": [
                    round((m["device_free_delta_mib"] - m["torch_reserved_delta_mib"]) / m["copies"], 3)
                    for _, m in records
                ],
                "capture_seconds_median": [
                    statistics.median(m["capture_seconds"]) for _, m in records
                ],
            }
            for arm, records in sorted(arms.items())
        }
    for shape, records in sorted(runs.items()):
        limit = records[0][1]["limit"]
        seqs = records[0][1]["max_num_seqs"]
        per_n = {}
        for n in range(1, limit + 1):
            key = str(n)
            row = {}
            for mode in ("hot_us", "cold_us"):
                deltas = [
                    statistics.fmean(r["replay"]["ladder"][key][mode])
                    - statistics.fmean(r["replay"]["base"][key][mode])
                    for _, r in records
                ]
                base = [statistics.fmean(r["replay"]["base"][key][mode]) for _, r in records]
                mean, half = interval(deltas)
                row[mode] = {
                    "base": round(statistics.fmean(base), 2),
                    "delta": round(mean, 2),
                    "ci95": round(half, 2),
                }
            row["capacity"] = {
                arm: records[0][1]["replay"][arm][key]["capacity"] for arm in ("base", "ladder")
            }
            per_n[n] = row

        def average(ns, mode):
            deltas = [
                statistics.fmean(
                    statistics.fmean(r["replay"]["ladder"][str(n)][mode])
                    - statistics.fmean(r["replay"]["base"][str(n)][mode])
                    for n in ns
                )
                for _, r in records
            ]
            mean, half = interval(deltas)
            return {"delta": round(mean, 2), "ci95": round(half, 2)}

        full_depth = [6 * c for c in range(1, seqs + 1)]
        padded = [n for n in range(1, limit + 1) if per_n[n]["capacity"]["base"] != per_n[n]["capacity"]["ladder"]]
        # The same graph size under both ladders: an A/A comparison, the noise floor.
        unchanged = [n for n in range(1, limit + 1) if n not in padded]
        capture = {}
        for arm in ("base", "ladder"):
            first = [r["capture"][arm] for _, r in records if r["order"][0] == arm]
            capture[arm] = {
                "graphs": first[0]["graphs"] if first else None,
                "capture_seconds": [c["capture_seconds"] for c in first],
                "device_free_delta_mib": [c["device_free_delta_mib"] for c in first],
                "torch_reserved_delta_mib": [c["torch_reserved_delta_mib"] for c in first],
            }
        summary[shape] = {
            "runs": [name for name, _ in records],
            "bit_mismatch_rows": sorted({n for _, r in records for n in r["bit_mismatch_rows"]}),
            "capture_first": capture,
            "full_depth": {
                f"c{n // 6}": {"rows": n, **per_n[n]} for n in full_depth
            },
            "average": {
                mode: {
                    "all_rows_1_to_limit": average(range(1, limit + 1), mode),
                    "rows_whose_capacity_changes": average(padded, mode),
                    "rows_whose_capacity_is_unchanged": average(unchanged, mode),
                    "full_depth_c1_to_cmax": average(full_depth, mode),
                    "rows_1_to_12": average(range(1, 13), mode),
                }
                for mode in ("hot_us", "cold_us")
            },
            "per_rows": per_n,
        }
    json.dump(summary, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "runs")
