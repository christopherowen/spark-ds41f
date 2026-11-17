"""Per-shape grid split of the four screen boots (two learner off, two on) and against candidate-60's recorded screen; run from the repository root."""
import json, statistics, math, sys
S = "experiments/2026-11-17-dspark-turn-learning/receipts/offstep-screen/"
R = "experiments/2026-11-16-c60-prefill/receipts/final/tp4-switched/suites/c60-screen-tp4-switched/"
def load(p):
    return {r["label"]: r for r in json.load(open(p))["engine"]}
arms = {a: load(S + a + "/grid.json") for a in ["off-1", "off-2", "offstep-1", "offstep-2"]}
arms["c60"] = load(R + "c60/grid.json"); arms["c60-again"] = load(R + "c60-again/grid.json")
labels = list(arms["off-1"])
def m(a, l, k):
    v = arms[a][l].get(k)
    return v["mean"] if v else None
def gm(a, k):
    return math.exp(statistics.mean(math.log(m(a, l, k)) for l in labels))
print("geometric mean step ms:", {a: round(gm(a, "step_ms"), 4) for a in arms})
print("geometric mean forward ms:", {a: round(gm(a, "forward_ms"), 4) for a in arms})
print("geometric mean drafter ms:", {a: round(gm(a, "drafter_ms"), 4) for a in arms})
def pair(A, B, k):  # per-shape delta, mean of B boots minus mean of A boots, µs
    out = []
    for l in labels:
        a = [m(x, l, k) for x in A]; b = [m(x, l, k) for x in B]
        out.append(1000 * (statistics.mean(b) - statistics.mean(a)))
    return out
def pct(A, B):
    return 100 * (math.exp(statistics.mean(math.log(gm(b, "step_ms")) for b in B)) / math.exp(statistics.mean(math.log(gm(a, "step_ms")) for a in A)) - 1)
for name, A, B in [("off -> offstep (same session)", ["off-1", "off-2"], ["offstep-1", "offstep-2"]),
                   ("off-1 -> off-2 (repeat)", ["off-1"], ["off-2"]),
                   ("offstep-1 -> offstep-2 (repeat)", ["offstep-1"], ["offstep-2"]),
                   ("c60 recorded -> off (drift)", ["c60", "c60-again"], ["off-1", "off-2"]),
                   ("c60 recorded -> offstep", ["c60", "c60-again"], ["offstep-1", "offstep-2"])]:
    s, f, d = pair(A, B, "step_ms"), pair(A, B, "forward_ms"), pair(A, B, "drafter_ms")
    print(f"{name:34s} step gm {pct(A,B):+.2f}%  median Δ step {statistics.median(s):+6.1f} µs  forward {statistics.median(f):+6.1f} µs  drafter {statistics.median(d):+6.1f} µs")
print()
print(f"{'shape':24s}" + "".join(f"{a:>11s}" for a in ["off-1","off-2","offstep-1","offstep-2"]) + f"{'Δ step µs':>11s}{'Δ drafter':>10s}")
for l in labels:
    st = [m(a, l, "step_ms") for a in ["off-1","off-2","offstep-1","offstep-2"]]
    dr = [m(a, l, "drafter_ms") for a in ["off-1","off-2","offstep-1","offstep-2"]]
    print(f"{l:24s}" + "".join(f"{x:11.3f}" for x in st) + f"{1000*((st[2]+st[3])-(st[0]+st[1]))/2:+11.1f}{1000*((dr[2]+dr[3])-(dr[0]+dr[1]))/2:+10.1f}")
