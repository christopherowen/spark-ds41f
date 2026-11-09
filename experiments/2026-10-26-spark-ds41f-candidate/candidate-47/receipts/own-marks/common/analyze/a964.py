import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/bench-{r}.json")) for r in runs}
print("all_bit_exact", {r: d[r]["all_bit_exact"] for r in runs})
for tp in ("TP3_h24", "TP4_h16"):
    for ctx in ("med_ctx_128_512", "short_ctx_128_64"):
        print(tp, ctx, " image-kernel(base) mean us: 1107 b1 b2 | c47 b1 b2 | xdelta | in-image s1-base: 1107 b1 b2 | c47 b1 b2")
        for M in ["1", "2", "4", "6", "8", "12", "16", "24", "32", "48"]:
            x = {r: d[r]["shapes"][tp][ctx][M] for r in runs}
            b = [x["1107-boot1"]["base_1107"]["mean_us"], x["1107-boot2"]["base_1107"]["mean_us"]]
            c = [x["c47-boot1"]["base_1107"]["mean_us"], x["c47-boot2"]["base_1107"]["mean_us"]]
            ds = [x[r]["delta_s1_us"] for r in ("1107-boot1", "1107-boot2", "c47-boot1", "c47-boot2")]
            sd = [x[r]["base_1107"]["stdev_us"] for r in runs]
            print(f"  M{M:>3} {x['1107-boot1']['route']:>6} {b[0]:7.2f} {b[1]:7.2f} | {c[0]:7.2f} {c[1]:7.2f} | {sum(c)/2-sum(b)/2:+6.2f} ({max(c)-min(b):+6.2f}) | " + " ".join(f"{v:+6.2f}" for v in ds) + f" | sd {max(sd):.2f}")
