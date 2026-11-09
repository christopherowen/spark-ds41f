import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/h1016-{r}.json")) for r in runs}
print("TC set:", {r: d[r]["image_tile_contiguous_fp8"] for r in ("1107-boot1", "c47-boot1")})
neq = {r: [k for k, v in d[r]["image_cfg_vs_old1107"].items() if not v["old1107_equal"]] for r in ("1107-boot1", "c47-boot1")}
print("image cfg != OLD_1107:", neq)
modes = ("eager_warm", "graph_warm", "eager_cold", "graph_cold", "eager_next", "graph_next")
for lab in ("tp4_8192_plus_4096", "tp3_12288_plus_4096"):
    print(lab)
    for M in ["1", "2", "3", "6", "12", "24", "48"]:
        rows = {r: d[r]["chained"][lab][M] for r in runs}
        bits = {r: rows[r]["bits_sha256_16"]["image"] for r in runs}
        eq = all(rows[r]["bit_equal_old1107_vs_image"] for r in runs)
        print(f"  M{M:>3} bits-same-across-images {len(set(bits.values()))==1} old==image-in-run {eq}")
        for mo in modes:
            b = [rows[r]["image"][mo]["mean"] for r in ("1107-boot1", "1107-boot2")]
            c = [rows[r]["image"][mo]["mean"] for r in ("c47-boot1", "c47-boot2")]
            ci = [rows[r]["image"][mo]["ci95"] for r in runs]
            pr = [rows[r]["paired_image_minus_old1107_us"][mo] for r in ("c47-boot1", "c47-boot2")]
            pa = [rows[r]["paired_image_minus_old1107_us"][mo] for r in ("1107-boot1", "1107-boot2")]
            print(f"     {mo:>10} 1107 {b[0]:7.2f} {b[1]:7.2f} | c47 {c[0]:7.2f} {c[1]:7.2f} | x-img {sum(c)/2-sum(b)/2:+6.2f} (ci<= {max(ci):.2f}) | c47 paired img-old {pr[0]['mean']:+6.2f}±{pr[0]['ci95']:.2f} {pr[1]['mean']:+6.2f}±{pr[1]['ci95']:.2f} | 1107 A/A {pa[0]['mean']:+6.2f}±{pa[0]['ci95']:.2f} {pa[1]['mean']:+6.2f}±{pa[1]['ci95']:.2f}")
print("layer chain (image TC set; graph replay)")
for k in d["1107-boot1"]["layer_chain"]:
    x = {r: d[r]["layer_chain"][k] for r in runs}
    dig = len(set(x[r]["digests_sha256"] for r in runs)) == 1
    for mo in ("graph_warm_us", "graph_cold_us"):
        b = [x[r][mo]["mean"] for r in ("1107-boot1", "1107-boot2")]; c = [x[r][mo]["mean"] for r in ("c47-boot1", "c47-boot2")]
        print(f"  {k:>8} {mo} 1107 {b[0]:8.2f}±{x['1107-boot1'][mo]['ci95']:.2f} {b[1]:8.2f}±{x['1107-boot2'][mo]['ci95']:.2f} | c47 {c[0]:8.2f}±{x['c47-boot1'][mo]['ci95']:.2f} {c[1]:8.2f}±{x['c47-boot2'][mo]['ci95']:.2f} | {sum(c)/2-sum(b)/2:+7.2f} | digests equal across images {dig}")
