import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/bench-{r}.json")) for r in runs}
for lab in ("tp4_8192_plus_4096", "tp3_12288_plus_4096"):
    print(lab, "verbatim bench (eager, trimmed means): per mode  bench-1107-arm(c47 b1,b2) | tc_aligned on 1107 image (b1,b2) | tc_aligned on c47 (b1,b2)")
    for M in ["1", "2", "3", "6", "12", "24", "48"]:
        out = []
        for mo in ("warm_us", "cold_us", "next_us"):
            x = {r: d[r]["chained_qb_plus_indexer_qb"][lab][M][mo] for r in runs}
            out.append(f"{mo[:4]} old {x['c47-boot1']['1107']:.2f},{x['c47-boot2']['1107']:.2f} img1107 {x['1107-boot1']['tc_aligned']:.2f},{x['1107-boot2']['tc_aligned']:.2f} c47 {x['c47-boot1']['tc_aligned']:.2f},{x['c47-boot2']['tc_aligned']:.2f}")
        print(f"  M{M:>3} " + " | ".join(out))
print("layer_chain_pdl (graph): tc on 1107 image vs tc on c47; rm both")
for k in d["c47-boot1"]["layer_chain_pdl"]:
    x = {r: d[r]["layer_chain_pdl"][k] for r in runs}
    for mo in ("warm_us", "cold_us"):
        print(f"  {k} {mo} tc: 1107 {x['1107-boot1'][mo]['tc_aligned']},{x['1107-boot2'][mo]['tc_aligned']} c47 {x['c47-boot1'][mo]['tc_aligned']},{x['c47-boot2'][mo]['tc_aligned']} | rm: 1107 {x['1107-boot1'][mo]['rm']},{x['1107-boot2'][mo]['rm']} c47 {x['c47-boot1'][mo]['rm']},{x['c47-boot2'][mo]['rm']} digests_equal {[x[r]['digests_equal'] for r in runs]}")
