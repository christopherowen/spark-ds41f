"""Where a step's extra time builds up: the step's stream's kernels aligned by their index in the step,
each kernel's end time from the step's first kernel (mean over steps), the candidate minus the base,
summed into segments between consecutive all-reduces."""
import collections, gzip, json, statistics, sys

def steps(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    s = collections.Counter(e["args"].get("stream") for e in k).most_common(1)[0][0]
    k = sorted((e for e in k if e["args"].get("stream") == s), key=lambda e: e["ts"])
    first = k[0]["name"]
    # a step starts at each occurrence of the step's first kernel name at the stream's position 0's period
    n = len(k)
    period = None
    for p in range(100, 5000):
        if n % p == 0 and all(k[i]["name"] == k[i % p]["name"] for i in range(0, n, 97)):
            period = p; break
    return [k[i:i + period] for i in range(0, n, period)], period

(bs, p), (cs, q) = steps(sys.argv[1]), steps(sys.argv[2])
assert p == q
print(f"{len(bs)} steps of {p} kernels on the step's stream")
skip = 4  # warm-up runs
def ends(ss):
    return [statistics.mean(st[i]["ts"] + st[i]["dur"] - st[0]["ts"] for st in ss[skip:]) for i in range(p)]
eb, ec = ends(bs), ends(cs)
print(f"step (first kernel start to last kernel end): {eb[-1]:.1f} -> {ec[-1]:.1f} µs ({ec[-1]-eb[-1]:+.1f})")
names = [x["name"] for x in bs[0]]
# segments: from one all-reduce's end to the next all-reduce's end
marks = [i for i, nm in enumerate(names) if nm.startswith("oneshot_allreduce")]
segs = []
prev = 0
for m in marks + [p - 1]:
    d = (ec[m] - ec[prev]) - (eb[m] - eb[prev])
    segs.append((d, prev, m))
    prev = m
print(f"{len(marks)} all-reduces a step; segment growth (µs) between consecutive all-reduce ends:")
for d, a, b in segs[:8]:
    print(f"  {a:5d}..{b:5d} {d:+7.1f}  ({names[a][:24]} .. {names[b][:24]})")
print("  ...")
tot = collections.defaultdict(float)
for d, a, b in segs:
    key = tuple(nm[:20] for nm in names[a + 1:b + 1][:3])
    tot[key] += d
print("by segment kind (first three kernels after the segment's start):")
for key, d in sorted(tot.items(), key=lambda kv: -abs(kv[1]))[:8]:
    print(f"  {d:+8.1f} µs  {key}")
# within-step: the largest single-kernel contributions (end-to-end delta minus previous kernel's)
inc = sorted(((ec[i] - ec[i - 1]) - (eb[i] - eb[i - 1]), i) for i in range(1, p))
print("largest single-kernel growth in end time:")
agg = collections.defaultdict(float)
for d, i in inc:
    agg[names[i][:40]] += d
for nm, d in sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:10]:
    print(f"  {d:+8.1f} µs a step  {nm}")
