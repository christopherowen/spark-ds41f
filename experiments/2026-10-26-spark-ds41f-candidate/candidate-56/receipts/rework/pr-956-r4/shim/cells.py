"""Bench cells from the ABBA runs: per (layer, T) and statistic, each boot's value, each
arm's mean of its boot values, the change, its 95% interval (Student t on
the boot medians, pooled variance, df = boots_a + boots_b - 2) and whether the boots
separate (every candidate boot below every base boot).

Two statistics per boot: "all_ranks" is the median over every timed forward (the four
ranks alternate); "slowest_rank" is the largest of the four ranks' medians. A rank's
indexer scores only its own rows, and later rows see more context, so the last rank does
the most work; an SP step waits for its slowest rank at the next collective.

usage: cells.py <json dir> <out.json>
"""

import glob
import json
import math
import os
import statistics
import sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
src, dst = sys.argv[1], sys.argv[2]
runs = {}
for path in sorted(glob.glob(os.path.join(src, "bench-*-boot*.json"))):
    d = json.load(open(path))
    runs[os.path.basename(path)[:-5]] = d
by = {}
for name, d in runs.items():
    for r in d["results"]:
        key = (r["layer"], r["T"])
        by.setdefault(key, {}).setdefault(d["arm"], []).append(
            dict(run=name, all_ranks=r["ms_median"],
                 slowest_rank=max(r["ms_per_rank_median"].values()),
                 mean=r["ms_mean"], stdev=r["ms_stdev"],
                 n=len(r["ms_all"]), bytes_per_row=r["gathered_bytes_per_row"],
                 converged=r["converged"]))
cells = []
for (layer, T), arms in sorted(by.items()):
    a, b = arms.get("base", []), arms.get("cand", [])
    if not a or not b:
        continue
    for stat in ("all_ranks", "slowest_rank"):
        ma = [x[stat] for x in a]
        mb = [x[stat] for x in b]
        mean_a, mean_b = statistics.fmean(ma), statistics.fmean(mb)
        df = len(ma) + len(mb) - 2
        if df > 0:
            sp2 = ((len(ma) - 1) * (statistics.variance(ma) if len(ma) > 1 else 0)
                   + (len(mb) - 1) * (statistics.variance(mb) if len(mb) > 1 else 0)) / df
            half = T975.get(df, 1.96) * math.sqrt(sp2 * (1 / len(ma) + 1 / len(mb)))
        else:
            half = float("nan")
        delta = mean_b - mean_a
        cells.append(dict(
            layer=layer, T=T, statistic=stat,
            base_boot_medians_ms=[round(x, 3) for x in ma],
            cand_boot_medians_ms=[round(x, 3) for x in mb],
            base_ms=round(mean_a, 3), cand_ms=round(mean_b, 3),
            delta_ms=round(delta, 3),
            delta_pct=round(100 * delta / mean_a, 2),
            ci95_ms=[round(delta - half, 3), round(delta + half, 3)],
            ci95_pct=[round(100 * (delta - half) / mean_a, 2), round(100 * (delta + half) / mean_a, 2)],
            boots_separate=max(mb) < min(ma),
            gathered_bytes_per_row=dict(base=a[0]["bytes_per_row"], cand=b[0]["bytes_per_row"]),
            iterations_per_boot=a[0]["n"],
            converged=all(x["converged"] for x in a + b),
        ))
json.dump(dict(runs=sorted(runs), cells=cells), open(dst, "w"), indent=1)
for c in cells:
    print(f"layer {c['layer']:>2} T {c['T']:>6} {c['statistic']:>12}: base {c['base_ms']:8.3f} {c['base_boot_medians_ms']} "
          f"cand {c['cand_ms']:8.3f} {c['cand_boot_medians_ms']}  {c['delta_pct']:+.2f}% "
          f"CI {c['ci95_pct']} sep={c['boots_separate']} bytes/row {c['gathered_bytes_per_row']}")
