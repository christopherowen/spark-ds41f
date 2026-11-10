"""Per-layer kernel time of the profiled steps (nsys sqlite, graph nodes): the window's 1x2 and 1x6
steps together (the same mix in both arms; 9 complete steps each).

usage: python3 alayer.py BASE.sqlite CAND.sqlite
A step is the kernels between two LM-head launches (vocab_packed_projection_kernel); a layer
starts at each attention_front_kernel. Main-stream kernels only (L2 prefetch kernels are
reported separately). Prints, per layer, the median over steps of the main kernels' summed
time, wall span (first start to next layer's first start) and the prefetch kernels' time,
for both arms; the owner layers (2, 8, 14, 20) first.
"""
import sqlite3, statistics, sys, json

def layers(path):
    db = sqlite3.connect(path)
    S = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = [(s, e, S[n]) for s, e, n in db.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    heads = [i for i, k in enumerate(ks) if k[2] == "vocab_packed_projection_kernel"]
    steps = []
    for a, b in zip(heads, heads[1:]):
        seg = ks[a + 1:b]
        fronts = [i for i, k in enumerate(seg) if k[2] == "attention_front_kernel"]
        if len(fronts) < 40:
            continue
        per = []
        for li, (x, y) in enumerate(zip(fronts, fronts[1:] + [len(seg)])):
            lk = seg[x:y]
            main = [k for k in lk if k[2] != "l2_prefetch_kernel"]
            pf = [k for k in lk if k[2] == "l2_prefetch_kernel"]
            nxt = seg[y][0] if y < len(seg) else lk[-1][1]
            byname = {}
            for k in main:
                byname[k[2]] = byname.get(k[2], 0) + (k[1] - k[0]) / 1e3
            per.append({"main_us": sum(k[1] - k[0] for k in main) / 1e3, "span_us": (nxt - lk[0][0]) / 1e3,
                        "pf_us": sum(k[1] - k[0] for k in pf) / 1e3, "byname": byname})
        steps.append(per)
    return steps

b, c = layers(sys.argv[1]), layers(sys.argv[2])
print("steps", len(b), len(c), "layers", len(b[0]), len(c[0]))
def med(st, li, key):
    return statistics.median(s[li][key] for s in st)
rows = []
for li in range(min(len(b[0]), len(c[0]))):
    rows.append({"layer": li, **{f"{k}_{a}": round(med(st, li, k), 1) for k in ("span_us", "main_us", "pf_us") for a, st in (("base", b), ("cand", c))}})
own = [r for r in rows if r["layer"] in (2, 8, 14, 20)]
oth = [r for r in rows if r["layer"] not in (2, 8, 14, 20) and r["layer"] < 40]
for r in own: print("owner", r)
for r in [x for x in rows if x["layer"] in (1, 3, 9, 15, 21, 25)]: print("other", r)
def tot(rs, k): return round(sum(r[k] for r in rs), 1)
print("owner layers sum: span", tot(own, "span_us_base"), "->", tot(own, "span_us_cand"), " main", tot(own, "main_us_base"), "->", tot(own, "main_us_cand"), " pf", tot(own, "pf_us_base"), "->", tot(own, "pf_us_cand"))
print("other layers sum: span", tot(oth, "span_us_base"), "->", tot(oth, "span_us_cand"), " main", tot(oth, "main_us_base"), "->", tot(oth, "main_us_cand"), " pf", tot(oth, "pf_us_base"), "->", tot(oth, "pf_us_cand"))
# owner-layer kernel deltas
names = set()
for st in (b, c):
    for s in st:
        for li in (2, 8, 14, 20):
            names |= set(s[li]["byname"])
d = []
for n in names:
    mb = sum(statistics.median(s[li]["byname"].get(n, 0) for s in b) for li in (2, 8, 14, 20))
    mc = sum(statistics.median(s[li]["byname"].get(n, 0) for s in c) for li in (2, 8, 14, 20))
    d.append((round(mc - mb, 1), round(mb, 1), round(mc, 1), n))
for x in sorted(d, key=lambda x: -abs(x[0]))[:14]: print("owner kernels", x)
json.dump({"rows": rows}, open(sys.argv[3], "w"), indent=1) if len(sys.argv) > 3 else None
