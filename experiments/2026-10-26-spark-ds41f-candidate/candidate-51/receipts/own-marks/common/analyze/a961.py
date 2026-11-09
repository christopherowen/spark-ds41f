# usage: python3 a961.py JSON_DIR OUT.json
# Reads l2pf-{base,c51}-boot{1,2}.json (bench_l2_prefetch_ab.py at 9362f944).
# The bench's cand_tl_64k_p200ns / cand_cute_64k_p200ns arms are the image's own
# kernels: the base arm's are the base's 4 KiB grid-2 prefetch (VLLM_L2_PREFETCH_GRID=2),
# the candidate arm's the paced 64 KiB prefetch. off and orig_* are embedded in the script.
#  image_ab:       base arm cand_tl -> c51 arm cand_tl (the serving TileLang kernel)
#  in_process_c51: c51 arm orig_tp4_g2_4k -> cand_tl (the PR's own comparison)
import json
import sys

from common import fmt, load, verdict

d = sys.argv[1]
R = {(a, k): load(f"{d}/l2pf-{a}-boot{k}.json") for a in ("base", "c51") for k in (1, 2)}
out = {"standalone": {}, "chain": {}, "counts": {}}


def both(arm, get):
    return [get(R[(arm, 1)]), get(R[(arm, 2)])]


sa = lambda r, v, f: r["standalone_16mb"]["variants"][v][f]["median_us"]  # noqa: E731
for f in ("read", "total", "prefetch"):
    out["standalone"][f"image_ab:{f}"] = fmt(both("base", lambda r: sa(r, "cand_tl_64k_p200ns", f)),
                                             both("c51", lambda r: sa(r, "cand_tl_64k_p200ns", f)))
    out["standalone"][f"image_ab_cute:{f}"] = fmt(both("base", lambda r: sa(r, "cand_cute_64k_p200ns", f)),
                                                  both("c51", lambda r: sa(r, "cand_cute_64k_p200ns", f)))
    out["standalone"][f"in_process_c51:{f}"] = fmt(both("c51", lambda r: sa(r, "orig_tp4_g2_4k", f)),
                                                   both("c51", lambda r: sa(r, "cand_tl_64k_p200ns", f)))
    out["standalone"][f"control_off_base_vs_c51:{f}"] = fmt(both("base", lambda r: sa(r, "off", f)),
                                                            both("c51", lambda r: sa(r, "off", f)))
    out["standalone"][f"control_orig_g2_base_vs_c51:{f}"] = fmt(both("base", lambda r: sa(r, "orig_tp4_g2_4k", f)),
                                                                both("c51", lambda r: sa(r, "orig_tp4_g2_4k", f)))
hit = lambda r, v: r["standalone_16mb"]["variants"][v]["effective_l2_hit_pct"]  # noqa: E731
out["standalone"]["l2_hit_pct"] = {
    f"{a}-boot{k}": {v: hit(R[(a, k)], v) for v in R[(a, k)]["standalone_16mb"]["variants"]} for (a, k) in R}
out["standalone"]["hot_read_median_us"] = {f"{a}-boot{k}": R[(a, k)]["standalone_16mb"]["hot_read_median_us"] for (a, k) in R}

for mode in ("pure_wait", "mem_active"):
    out["chain"][mode] = {}
    for rk in R[("c51", 1)]["heterogeneous_4layer_chain"][mode]:
        g = lambda r, v: r["heterogeneous_4layer_chain"][mode][rk][v]["median_us"]  # noqa: E731
        row = {}
        for comp, b, c in (
            ("image_ab", both("base", lambda r: g(r, "cand_tl_64k_p200ns")), both("c51", lambda r: g(r, "cand_tl_64k_p200ns"))),
            ("in_process_c51", both("c51", lambda r: g(r, "orig_tp4_g2_4k")), both("c51", lambda r: g(r, "cand_tl_64k_p200ns"))),
            ("control_off", both("base", lambda r: g(r, "off")), both("c51", lambda r: g(r, "off"))),
            ("c51_off_vs_cand_tl", both("c51", lambda r: g(r, "off")), both("c51", lambda r: g(r, "cand_tl_64k_p200ns"))),
        ):
            row[comp] = fmt(b, c)
            v = verdict(b, c)
            out["counts"].setdefault(f"{mode}:{comp}", {}).setdefault(v, 0)
            out["counts"][f"{mode}:{comp}"][v] += 1
        out["chain"][mode][rk] = row
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
