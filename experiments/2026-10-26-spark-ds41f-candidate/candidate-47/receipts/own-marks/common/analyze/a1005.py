import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/bench-{r}.json")) for r in runs}
print("standalone moe_swiglu_cast graph us: base_copy/image_kernel per run (ABBA)")
for inter in ["576", "768", "1152", "1536"]:
    for M in ["1", "2", "3", "6", "12", "24", "48"]:
        s = d
        print(f"{inter:>5} M{M:>3} " + "  ".join(f"{s[r]['standalone_moe_swiglu_cast'][inter][M]['baseline_1107_graph_us']:.3f}/{s[r]['standalone_moe_swiglu_cast'][inter][M]['candidate_graph_us']:.3f}" for r in runs))
for tp in ["tp4_routed576_shared1152", "tp3_routed768_shared1536"]:
    for kind in ["warm_graph_us", "cold_l2pf_graph_us"]:
        print(tp, kind, "   [serial | base1107 | swiglu_only | cand_full] per run")
        for M in ["1", "2", "3", "6", "12", "24", "48"]:
            parts = []
            for r in runs:
                x = d[r]["full_ffn_sublayer"][tp][M]
                parts.append(f"{x['serial_1stream_'+kind]:.1f}|{x['baseline_1107_'+kind]:.1f}|{x['candidate_swiglu_only_'+kind]:.1f}|{x['candidate_full_'+kind]:.1f}")
            print(f"  M{M:>3} " + "  ".join(parts))
