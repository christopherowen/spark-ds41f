import json, sys
a = json.load(open(sys.argv[1]))
for topo, t in a["topologies"].items():
    print("==", topo, "warm>=", t["gpu_warm_ms_min"], "ms")
    for op, rows in t["ops"].items():
        for T, c in rows.items():
            A = c["arms"]
            s = f"{op[:7]} T={T:>5} " + " ".join(f"{k}={[round(x) for x in v['max_rank_medians_us']]}" for k, v in A.items())
            s += " | " + " ".join(f"{k[:5]} {v['change_pct']:+.1f}% {v['max_rank']}/{v['rank0']}({v['rank0_change_pct']:+.1f}%){'=' if v['identical_code'] else ''}"
                                for k, v in c.items() if k.endswith("_vs_base0"))
            s += f" | exact={c['bit_exact_across_runs']}/{c['equal_to_in_process_anchor']}"
            print(s)
print("A/A identical code:", {k: v for k, v in a["identical_code_aa"].items() if k != "cells"})
for x in a["identical_code_aa"]["cells"]:
    print("  ", x)
print("A/A anchors:", {k: v for k, v in a["anchor_aa"].items() if k != "cells"})
