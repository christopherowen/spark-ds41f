"""Compare the attention harness runs: for every (world, scenario, step, layer), every
rank's output, top-k and owned-cache digests across the arms (base, cand) and boots.

usage: compare.py <json dir> <out.json> <run label>...   (labels like base-tp4-boot1)
"""

import json
import sys
from collections import defaultdict

src, dst, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = {lab: json.load(open(f"{src}/{lab}.json")) for lab in labels}
cells = defaultdict(dict)
gathers = defaultdict(dict)
problems = []
for lab, d in runs.items():
    for s in d["scenarios"]:
        key = (d["world"], lab.split("-tp")[1].split("-", 1)[1] if "breakable" in lab else "serving",
               s["scenario"], s["step"], s["layer"])
        if s["error"] or not s["converged"]:
            problems.append(dict(run=lab, scenario=s["scenario"], step=s["step"], layer=s["layer"],
                                 error=s["error"], converged=s["converged"]))
            continue
        for flag in ("ranks_agree_on_output", "ranks_agree_on_topk", "ranks_agree_on_caches"):
            if not s.get(flag):
                problems.append(dict(run=lab, scenario=s["scenario"], step=s["step"],
                                     layer=s["layer"], failed=flag))
        cells[key][lab] = [
            dict(out=r["out"], topk=r["topk"], caches=r["caches"], finite=r["out_finite"])
            for r in s["ranks"]
        ]
        gathers[key][d["arm"]] = [g[0] for g in s["ranks"][0]["gathers"]]
table = []
for key in sorted(cells, key=str):
    world, mode, scenario, step, layer = key
    by = cells[key]
    digests = {lab: json.dumps(v, sort_keys=True) for lab, v in by.items()}
    identical = len(set(digests.values())) == 1
    table.append(dict(world=world, mode=mode, scenario=scenario, step=step, layer=layer,
                      runs=sorted(by), identical_across_runs=identical,
                      all_finite=all(r["finite"] for v in by.values() for r in v),
                      out_rank0={lab: v[0]["out"] for lab, v in sorted(by.items())},
                      gathered_rows_shapes=gathers[key]))
summary = dict(
    runs=labels,
    cells=len(table),
    identical_cells=sum(t["identical_across_runs"] for t in table),
    problems=problems,
    table=table,
)
json.dump(summary, open(dst, "w"), indent=1)
print(f"{summary['identical_cells']}/{summary['cells']} cells identical across "
      f"{len(labels)} runs; {len(problems)} problems")
for t in table:
    if not t["identical_across_runs"]:
        print("DIFF", t["world"], t["mode"], t["scenario"], t["step"], t["layer"], t["out_rank0"])
for p in problems:
    print("PROBLEM", p)
