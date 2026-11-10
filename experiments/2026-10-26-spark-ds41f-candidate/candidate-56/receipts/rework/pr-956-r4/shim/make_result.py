"""Write round3-956-r4/result.json from the raw outputs (run on the laptop after the raw
folder is copied back). usage: make_result.py <round3-956-r4 dir>"""

import hashlib
import json
import re
import sys

D = sys.argv[1]
PATCH = "0796-perf-dsv41-sp-ratio2-compressor-and-ced-encoder-only-chunks.patch"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(p):
    return json.load(open(f"{D}/{p}"))


def tail(p, pat):
    try:
        lines = [l for l in open(f"{D}/{p}", errors="replace") if re.search(pat, l)]
        return lines[-1].strip() if lines else None
    except FileNotFoundError:
        return None


cells = load("raw/json/bench-cells.json")["cells"]
cmp4, cmp3, cmpb = (load(f"raw/json/harness-compare-{x}.json")
                    for x in ("serving-tp4", "serving-tp3", "breakable-tp4"))
bench = {n: load(f"raw/json/bench-{n}.json") for n in
         ("base-boot1", "cand-boot1", "cand-boot2", "base-boot2")}
mem = {n: b["layers"]["_memory"]["allocated_after_declare"] for n, b in bench.items()}
spw = {n: {k: v.get("sp_weight_bytes") for k, v in b["layers"].items() if k != "_memory"}
       for n, b in bench.items()}


def rows(stat, layer):
    return [dict(T=c["T"], base_ms=c["base_ms"], cand_ms=c["cand_ms"],
                 base_boots_ms=c["base_boot_medians_ms"], cand_boots_ms=c["cand_boot_medians_ms"],
                 delta_pct=c["delta_pct"], ci95_pct=c["ci95_pct"],
                 boots_separate=c["boots_separate"],
                 sp_gathered_bytes_per_local_row=c["gathered_bytes_per_row"])
            for c in cells if c["statistic"] == stat and c["layer"] == layer]


def hsum(c):
    return dict(runs=c["runs"], sp_cells=c["cells"], sp_identical=c["identical_cells"],
                grid_cells=c["grid_cells"], grid_identical=c["grid_identical"],
                grid_identical_with_per_stream_order=c["grid_identical_with_order"],
                problems=c["problems"],
                grid=[dict(scenario=t["scenario"], layer=t["layer"], kernels=t["kernels"],
                           identical=t["identical_outputs_caches_kernel_names"])
                      for t in c["grid"]])


result = json.load(open(f"{D}/result-skeleton.json"))
result["patch"]["sha256"] = sha(f"{D}/{PATCH}")
result["bit_exactness"]["attention_harness"]["tp4_serving_env"] = hsum(cmp4)
result["bit_exactness"]["attention_harness"]["tp3_serving_env"] = hsum(cmp3)
result["bit_exactness"]["attention_harness"]["tp4_breakable_graphs"] = hsum(cmpb)
result["decode_unchanged"]["harness_grid"] = dict(
    tp4=f"{cmp4['grid_identical']}/{cmp4['grid_cells']}",
    tp3=f"{cmp3['grid_identical']}/{cmp3['grid_cells']}",
    tp4_breakable=f"{cmpb['grid_identical']}/{cmpb['grid_cells']}",
    with_per_stream_launch_order=dict(tp4=cmp4["grid_identical_with_order"],
                                      tp3=cmp3["grid_identical_with_order"],
                                      tp4_breakable=cmpb["grid_identical_with_order"]))
result["own_mark"]["cells_layer2_all_ranks"] = rows("all_ranks", 2)
result["own_mark"]["cells_layer20_all_ranks"] = rows("all_ranks", 20)
result["own_mark"]["cells_slowest_rank"] = dict(layer2=rows("slowest_rank", 2),
                                                layer20=rows("slowest_rank", 20))
result["static_memory"]["measured_one_gpu"] = dict(
    allocated_after_declare_bytes=mem, sp_weight_bytes=spw,
    delta_cand_minus_base_bytes=sorted({mem[c] - mem[b] for c in ("cand-boot1", "cand-boot2")
                                        for b in ("base-boot1", "base-boot2")}))
for k, p in (("newtests_cand", "raw/pytest-newtests-cand.txt"),
             ("cand_boot1", "raw/pytest-cand-boot1.txt"),
             ("cand_boot2", "raw/pytest-cand-boot2.txt"),
             ("base_boot1", "raw/pytest-base-boot1.txt"),
             ("newtests_on_base_code", "raw/pytest-newtests-on-base.txt")):
    result["tests"][k] = tail(p, r"(passed|failed|error)")
result["apply_check"]["receipt_lines"] = [
    l.strip() for l in open(f"{D}/raw/apply-check.txt") if l.strip()]
json.dump(result, open(f"{D}/result.json", "w"), indent=1)
print("written")
