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
grid = defaultdict(dict)
for lab, d in runs.items():
    mode = lab.split("-tp")[1].split("-", 1)[1] if "breakable" in lab else "serving"
    for g in d.get("grid", []):
        key = (d["world"], mode, g["scenario"], g["layer"])
        if g.get("error"):
            problems.append(dict(run=lab, scenario=g["scenario"], layer=g["layer"],
                                 error=g["error"]))
            continue
        grid[key][lab] = dict(out=g["out"], topk=g["topk"], caches=g["caches"],
                              kernels=g["kernels"], names=g["kernel_names_digest"],
                              sequence=g["kernel_sequence_digest"],
                              finite=g["out_finite"])
grid_table = []
for key in sorted(grid, key=str):
    by = grid[key]
    values = {lab: json.dumps({k: v for k, v in x.items()}, sort_keys=True)
              for lab, x in by.items()}
    no_seq = {lab: json.dumps({k: v for k, v in x.items() if k != "sequence"},
                              sort_keys=True) for lab, x in by.items()}
    grid_table.append(dict(world=key[0], mode=key[1], scenario=key[2], layer=key[3],
                           runs=sorted(by),
                           identical_outputs_caches_kernel_names=len(set(no_seq.values())) == 1,
                           identical_including_per_stream_order=len(set(values.values())) == 1,
                           kernels={lab: x["kernels"] for lab, x in sorted(by.items())},
                           out_rank0={lab: x["out"] for lab, x in sorted(by.items())}))
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
    grid_cells=len(grid_table),
    grid_identical=sum(t["identical_outputs_caches_kernel_names"] for t in grid_table),
    grid_identical_with_order=sum(t["identical_including_per_stream_order"]
                                  for t in grid_table),
    problems=problems,
    table=table,
    grid=grid_table,
)
json.dump(summary, open(dst, "w"), indent=1)
print(f"{summary['identical_cells']}/{summary['cells']} cells identical across "
      f"{len(labels)} runs; grid {summary['grid_identical']}/{summary['grid_cells']} "
      f"identical (with per-stream order {summary['grid_identical_with_order']}); "
      f"{len(problems)} problems")
for t in grid_table:
    if not t["identical_outputs_caches_kernel_names"]:
        print("GRID DIFF", t["world"], t["mode"], t["scenario"], t["layer"], t["kernels"])
for t in table:
    if not t["identical_across_runs"]:
        print("DIFF", t["world"], t["mode"], t["scenario"], t["step"], t["layer"], t["out_rank0"])
for p in problems:
    print("PROBLEM", p)
