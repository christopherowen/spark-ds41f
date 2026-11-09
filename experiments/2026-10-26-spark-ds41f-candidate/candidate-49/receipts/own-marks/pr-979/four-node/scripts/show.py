import json, sys
a = json.load(open(sys.argv[1]))
for topo, t in a["topologies"].items():
    print("==", topo, t["runs"])
    for op, rows in t["ops"].items():
        for T, c in rows.items():
            print(f"{op[:7]} T={T:>5} c49={c['c49']['max_rank_medians_us']} base={c['no979']['max_rank_medians_us']} "
                  f"chg={c['change_pct']}% ({c['change_us']}us) {c['beyond_interval']} | r0 {c['rank0_change_pct']}% {c['rank0_beyond_interval']} "
                  f"| exact={c['bit_exact_across_runs']}/{c['equal_to_in_process_anchor']} | anchors c={c['c49']['anchor_max_rank_medians_us']} b={c['no979']['anchor_max_rank_medians_us']} | route={c['c49']['route']}")
