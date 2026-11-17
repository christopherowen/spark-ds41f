"""Gaps before each kernel on the step's stream, aligned kernel by kernel between two traces of the same
grid window (identical launch sequences): where the timeline's extra time sits."""
import collections, gzip, json, sys

def main_stream(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    stream = collections.Counter(e["args"].get("stream") for e in k).most_common(1)[0][0]
    k = sorted((e for e in k if e["args"].get("stream") == stream), key=lambda e: e["ts"])
    return k

base, cand = main_stream(sys.argv[1]), main_stream(sys.argv[2])
assert [e["name"] for e in base] == [e["name"] for e in cand], "launch sequences differ"
def gaps(k):
    return [0.0] + [k[i]["ts"] - (k[i - 1]["ts"] + k[i - 1]["dur"]) for i in range(1, len(k))]
gb, gc = gaps(base), gaps(cand)
db, dc = [e["dur"] for e in base], [e["dur"] for e in cand]
span = lambda k: (k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]) / 1000
print(f"step stream: {len(base)} kernels; span {span(base):.1f} -> {span(cand):.1f} ms; "
      f"kernel time {sum(db)/1000:.1f} -> {sum(dc)/1000:.1f} ms; gaps {sum(gb)/1000:.1f} -> {sum(gc)/1000:.1f} ms")
by_next = collections.defaultdict(lambda: [0.0, 0.0, 0])
by_dur = collections.defaultdict(lambda: [0.0, 0.0, 0])
for i, e in enumerate(base):
    prev = base[i - 1]["name"] if i else "-"
    key = (prev[:40], e["name"][:40])
    by_next[key][0] += gb[i]; by_next[key][1] += gc[i]; by_next[key][2] += 1
    by_dur[e["name"][:50]][0] += db[i]; by_dur[e["name"][:50]][1] += dc[i]; by_dur[e["name"][:50]][2] += 1
print("\nlargest gap changes, by (kernel before, kernel after) on the step's stream:")
for (a, b), (x, y, n) in sorted(by_next.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:12]:
    print(f"  {(y - x)/1000:+8.2f} ms  ({x/1000:7.2f} -> {y/1000:7.2f}, n={n})  {a} -> {b}")
print("\nlargest duration changes on the step's stream:")
for name, (x, y, n) in sorted(by_dur.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:8]:
    print(f"  {(y - x)/1000:+8.2f} ms  ({x/1000:7.2f} -> {y/1000:7.2f}, n={n})  {name}")
