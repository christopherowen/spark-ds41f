"""Per stream: kernel time, and how much of it ran while a kernel of the step's stream was running."""
import bisect, collections, gzip, json, sys

for path in sys.argv[1:]:
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    by = collections.defaultdict(list)
    for e in k:
        by[e["args"].get("stream")].append(e)
    main = max(by, key=lambda s: len(by[s]))
    # merged busy intervals of the step's stream
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in by[main])
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    starts = [m[0] for m in merged]
    def overlap(a, b):
        i = max(bisect.bisect_right(starts, a) - 1, 0); t = 0.0
        while i < len(merged) and merged[i][0] < b:
            t += max(0.0, min(b, merged[i][1]) - max(a, merged[i][0])); i += 1
        return t
    print(path.split("/")[-1])
    for s in sorted(by, key=lambda s: -len(by[s])):
        if s == main:
            continue
        tot = sum(e["dur"] for e in by[s]); ov = sum(overlap(e["ts"], e["ts"] + e["dur"]) for e in by[s])
        names = collections.Counter(e["name"][:28] for e in by[s]).most_common(2)
        print(f"  stream {s}: {len(by[s])} kernels, {tot/1000:.1f} ms, {100*ov/max(tot,1):.1f}% beside the step's stream; {names}")
