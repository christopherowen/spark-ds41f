import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/bench-{r}.json")) for r in runs}
out = {}
for tp in ["tp4_routed576_shared1152", "tp3_routed768_shared1536"]:
    out[tp] = {}
    for kind in ["warm_graph_us", "cold_l2pf_graph_us"]:
        rows = {}
        for M in ["1", "2", "3", "6", "12", "24", "48"]:
            g = lambda r, mode: d[r]["full_ffn_sublayer"][tp][M][f"{mode}_{kind}"]
            rows[M] = {
                "c47_cand_full_minus_base1107mode_b1_b2": [round(g(r, "candidate_full") - g(r, "baseline_1107"), 2) for r in ("c47-boot1", "c47-boot2")],
                "aa_1107_swiglu_only_minus_base_b1_b2": [round(g(r, "candidate_swiglu_only") - g(r, "baseline_1107"), 2) for r in ("1107-boot1", "1107-boot2")],
                "x_image_cand_full_c47_minus_base_1107": round((g("c47-boot1", "candidate_full") + g("c47-boot2", "candidate_full") - g("1107-boot1", "baseline_1107") - g("1107-boot2", "baseline_1107")) / 2, 2),
                "x_image_same_base_mode_c47_minus_1107_(0763)": round((g("c47-boot1", "baseline_1107") + g("c47-boot2", "baseline_1107") - g("1107-boot1", "baseline_1107") - g("1107-boot2", "baseline_1107")) / 2, 2),
                "base1107mode_1107_b1_b2": [g(r, "baseline_1107") for r in ("1107-boot1", "1107-boot2")],
                "cand_full_c47_b1_b2": [g(r, "candidate_full") for r in ("c47-boot1", "c47-boot2")],
            }
        out[tp][kind] = rows
print(json.dumps(out, indent=1))
