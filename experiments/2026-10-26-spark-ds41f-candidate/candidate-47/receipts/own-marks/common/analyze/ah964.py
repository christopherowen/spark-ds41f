import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/h964-{r}.json")) for r in runs}
allsame = True
for tp in ("TP3_h24", "TP4_h16"):
    for ctx in ("med_ctx_128_512", "short_ctx_128_64"):
        print(tp, ctx, "graph us image-kernel: 1107 b1 b2 | c47 b1 b2 | x-img | worst-c47 minus best-1107 (with ci) | in-image bench_s1-image: 1107 b1 b2 | c47 b1 b2 | bits")
        for M in ["1", "2", "4", "6", "8", "12", "16", "24", "32", "48"]:
            x = {r: d[r]["shapes"][tp][ctx][M] for r in runs}
            hs = set(x[r]["sha256_16"]["image"] for r in runs) | set(x[r]["sha256_16"]["bench_s1"] for r in runs)
            eq = all(x[r]["image_equal_bench_s1"] for r in runs)
            allsame &= (len(hs) == 1 and eq)
            g = {r: x[r]["graph_us"]["image"] for r in runs}
            b = [g["1107-boot1"]["mean"], g["1107-boot2"]["mean"]]; c = [g["c47-boot1"]["mean"], g["c47-boot2"]["mean"]]
            w = max(g[r]["mean"] + g[r]["ci95"] for r in ("c47-boot1", "c47-boot2")) - min(g[r]["mean"] - g[r]["ci95"] for r in ("1107-boot1", "1107-boot2"))
            p = [x[r]["paired_bench_s1_minus_image_us"]["mean"] for r in ("1107-boot1", "1107-boot2", "c47-boot1", "c47-boot2")]
            print(f"  M{M:>3} {x['1107-boot1']['route']:>6} {b[0]:6.2f} {b[1]:6.2f} | {c[0]:6.2f} {c[1]:6.2f} | {sum(c)/2-sum(b)/2:+6.2f} ({100*(sum(c)-sum(b))/sum(b):+5.1f}%) | {w:+6.2f} | " + " ".join(f"{v:+5.2f}" for v in p) + f" | {'same' if len(hs)==1 and eq else 'DIFF'}")
print("all outputs bit-identical across images and arms:", allsame)
