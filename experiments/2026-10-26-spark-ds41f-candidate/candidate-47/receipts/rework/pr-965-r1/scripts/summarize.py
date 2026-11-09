"""Summarize the #965 step harness (bench_step_fastpath.py) and the clean-apply microbench
across stacks (c47 = candidate-47's 0766, fix = corrected 0766) and boots."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
stacks, boots, conc = ("c47", "fix"), (1, 2), ("c1", "c8", "c16")
H = {(s, b): json.load(open(out / f"harness-{s}-b{b}.json")) for s in stacks for b in boots}
C = {(s, b): json.load(open(out / f"clean-apply-{s}-b{b}.json")) for s in stacks for b in boots}
res = {"harness": {}, "clean_apply_us": {}}
for metric in ("total_step_overhead", "pipelined_step_overhead", "staged_writes"):
    for c in conc:
        row = {}
        for s in stacks:
            base = [H[s, b]["workloads"][c]["baseline"][metric]["mean_us"] for b in boots]
            cand = [H[s, b]["workloads"][c]["candidate"][metric]["mean_us"] for b in boots]
            exact = [H[s, b]["workloads"][c]["bit_exact"] for b in boots]
            row[s] = {
                "baseline_mean_us_per_boot": base,
                "candidate_mean_us_per_boot": cand,
                "baseline_range": [min(base), max(base)],
                "candidate_range": [min(cand), max(cand)],
                "gain_beyond_interval": max(cand) < min(base),
                "mean_delta_us": round(sum(cand) / 2 - sum(base) / 2, 2),
                "bit_exact_per_boot": exact,
            }
        f, o = row["fix"]["candidate_range"], row["c47"]["candidate_range"]
        row["fix_vs_c47_candidate"] = {
            "delta_mean_us": round(sum(row["fix"]["candidate_mean_us_per_boot"]) / 2
                                   - sum(row["c47"]["candidate_mean_us_per_boot"]) / 2, 2),
            "ranges_overlap": not (f[1] < o[0] or o[1] < f[0]),
        }
        res["harness"].setdefault(metric, {})[c] = row
for c in conc:
    res["clean_apply_us"][c] = {
        s: [C[s, b]["rows"][c]["mean_us"] for b in boots] for s in stacks
    }
json.dump(res, sys.stdout, indent=1)
print()
for metric in ("total_step_overhead", "pipelined_step_overhead"):
    for c in conc:
        r = res["harness"][metric][c]
        print(metric, c, "c47 base", r["c47"]["baseline_mean_us_per_boot"], "cand", r["c47"]["candidate_mean_us_per_boot"],
              "| fix base", r["fix"]["baseline_mean_us_per_boot"], "cand", r["fix"]["candidate_mean_us_per_boot"],
              "| beyond:", r["c47"]["gain_beyond_interval"], r["fix"]["gain_beyond_interval"],
              "| fix-c47 cand", r["fix_vs_c47_candidate"], file=sys.stderr)
for c in conc:
    print("clean apply", c, res["clean_apply_us"][c], file=sys.stderr)
