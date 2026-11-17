"""Grid split of the attribution job's six boots (learner off, 150 MiB held from load, 350 MiB; two boots each); run from the repository root."""
import json, math, statistics

S = "experiments/2026-11-17-dspark-turn-learning/receipts/reserve/"
ARMS = ["off-1", "off-2", "reserve150-1", "reserve150-2", "reserve350-1", "reserve350-2"]
g = {a: {r["label"]: r for r in json.load(open(S + a + "/grid.json"))["engine"]} for a in ARMS}
labels = list(g["off-1"])


def m(a, l, k):
    return g[a][l][k]["mean"]


def gm(a, k):
    return math.exp(statistics.mean(math.log(m(a, l, k)) for l in labels))


def both(arms, k):
    return math.exp(statistics.mean(math.log(gm(a, k)) for a in arms))


for k in ("step_ms", "forward_ms", "drafter_ms"):
    print("geometric mean", k, {a: round(gm(a, k), 4) for a in ARMS})
for name, A, B in [("off repeat", ["off-1"], ["off-2"]), ("150 repeat", ["reserve150-1"], ["reserve150-2"]),
                   ("350 repeat", ["reserve350-1"], ["reserve350-2"]),
                   ("off -> 150 MiB", ["off-1", "off-2"], ["reserve150-1", "reserve150-2"]),
                   ("off -> 350 MiB", ["off-1", "off-2"], ["reserve350-1", "reserve350-2"]),
                   ("150 -> 350 MiB", ["reserve150-1", "reserve150-2"], ["reserve350-1", "reserve350-2"])]:
    d = {k: [1000 * (statistics.mean(m(b, l, k) for b in B) - statistics.mean(m(a, l, k) for a in A)) for l in labels]
         for k in ("step_ms", "forward_ms", "drafter_ms")}
    print(f"{name:18s} step {100 * (both(B, 'step_ms') / both(A, 'step_ms') - 1):+.2f}%  forward "
          f"{100 * (both(B, 'forward_ms') / both(A, 'forward_ms') - 1):+.2f}%  median per shape: step "
          f"{statistics.median(d['step_ms']):+6.1f} µs, forward {statistics.median(d['forward_ms']):+6.1f} µs, "
          f"draft step {statistics.median(d['drafter_ms']):+6.1f} µs")
print(f"\n{'shape':24s}{'off':>9s}{"150":>9s}{"350":>9s}{"150 µs":>10s}{"350 µs":>11s}")
for l in labels:
    o, c, s = (statistics.mean(m(a, l, "step_ms") for a in arms) for arms in
               (["off-1", "off-2"], ["reserve150-1", "reserve150-2"], ["reserve350-1", "reserve350-2"]))
    print(f"{l:24s}{o:9.3f}{c:9.3f}{s:9.3f}{1000 * (c - o):+10.1f}{1000 * (s - o):+11.1f}")
