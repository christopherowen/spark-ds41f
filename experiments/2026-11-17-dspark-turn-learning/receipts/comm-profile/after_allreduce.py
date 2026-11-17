"""Each step all-reduce on the step's stream and the kernel after it: durations and offsets (µs, means)."""
import collections, gzip, json, statistics, sys

def main_stream(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    s = collections.Counter(e["args"].get("stream") for e in k).most_common(1)[0][0]
    return sorted((e for e in k if e["args"].get("stream") == s), key=lambda e: e["ts"])

for path in sys.argv[1:]:
    k = main_stream(path)
    rows = collections.defaultdict(list)
    for i in range(len(k) - 1):
        a, b = k[i], k[i + 1]
        if a["name"].startswith("oneshot_allreduce"):
            ar_end = a["ts"] + a["dur"]
            rows[b["name"][:26]].append((a["dur"], b["ts"] - a["ts"], b["ts"] - ar_end, b["dur"], b["ts"] + b["dur"] - ar_end))
    print(path.split("/")[-1])
    for name, r in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        m = [statistics.mean(x[j] for x in r) for j in range(5)]
        print(f"  -> {name:26s} n={len(r):5d}  all-reduce {m[0]:6.1f}  next starts {m[1]:6.1f} after its start ({m[2]:+6.1f} after its end)  next {m[3]:6.1f}  next ends {m[4]:6.1f} after the all-reduce ends")
