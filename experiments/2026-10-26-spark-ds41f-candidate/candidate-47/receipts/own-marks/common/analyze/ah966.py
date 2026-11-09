import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/h966-{r}.json")) for r in runs}
for sh in ("TP3", "TP4"):
    s = {r: d[r]["shapes"][sh] for r in runs}
    print(sh, "arms", {r: s[r]["arms"] for r in runs}, "err1107:", s["1107-boot1"].get("explicit_policy_arms_error", "")[:90])
    hs = {M: set(s[r]["bit_exact"][M]["sha256_16"]["default"] for r in runs) for M in s["1107-boot1"]["bit_exact"]}
    inimg = all(s[r]["bit_exact"][M]["equal_in_image"] for r in runs for M in s[r]["bit_exact"])
    print("  default-arm output hash identical across images per M:", {M: len(v) == 1 for M, v in hs.items()}, "in-image equal:", inimg)
    for sec in ("ffn", "moe"):
        print(f"  {sec}: mode | x-img default (1107 b1,b2 -> c47 b1,b2 means) | c47 in-image paired first-normal (b1, b2)")
        for M in s["c47-boot1"][sec]:
            for mo in ("eager", "graph", "chain16_graph"):
                b = [s[r][sec][M]["default"][mo] for r in ("1107-boot1", "1107-boot2")]
                c = [s[r][sec][M]["default"][mo] for r in ("c47-boot1", "c47-boot2")]
                p = [s[r][sec][M]["paired_first_minus_normal_us"][mo] for r in ("c47-boot1", "c47-boot2")]
                w = max(x["mean"] + x["ci95"] for x in c) - min(x["mean"] - x["ci95"] for x in b)
                print(f"    M{M:>3} {mo:>13} {b[0]['mean']:8.2f}±{b[0]['ci95']:.2f} {b[1]['mean']:8.2f}±{b[1]['ci95']:.2f} -> {c[0]['mean']:8.2f}±{c[0]['ci95']:.2f} {c[1]['mean']:8.2f}±{c[1]['ci95']:.2f} | {(c[0]['mean']+c[1]['mean']-b[0]['mean']-b[1]['mean'])/2:+7.2f} ({100*(c[0]['mean']+c[1]['mean']-b[0]['mean']-b[1]['mean'])/(b[0]['mean']+b[1]['mean']):+5.1f}%) worst {w:+7.2f} | paired {p[0]['mean']:+7.2f}±{p[0]['ci95']:.2f} {p[1]['mean']:+7.2f}±{p[1]['ci95']:.2f}")
