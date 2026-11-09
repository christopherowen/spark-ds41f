import json, sys
P = sys.argv[1]
for co in ("1", "0"):
    runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
    d = {}
    for r in runs:
        for line in open(f"{P}/bench-carveout{co}-{r}.txt"):
            if line.startswith("BENCH_JSON:"):
                d[r] = json.loads(line[len("BENCH_JSON:"):])
    print(f"carveout={co}")
    for shape in ("tp4-ring", "tp3-triangle"):
        print(" ", shape, "mismatches", {r: (d[r]["shapes"][shape]["mismatches"], d[r]["shapes"][shape]["total_rows_checked"]) for r in runs})
        for m in ["1", "2", "4", "6", "8", "16", "32", "48"]:
            v = {r: d[r]["shapes"][shape]["latencies_us"][m] for r in runs}
            b = [v["1107-boot1"], v["1107-boot2"]]; c = [v["c47-boot1"], v["c47-boot2"]]
            ac = d["c47-boot1"]["shapes"][shape]["active_cols"].get(m)
            print(f"    M{m:>3} 1107 {b[0]:8.2f} {b[1]:8.2f} | c47 {c[0]:8.2f} {c[1]:8.2f} | delta(mean) {sum(c)/2-sum(b)/2:+8.2f} | worst-c47 minus best-1107 {max(c)-min(b):+8.2f} | active {ac}")
