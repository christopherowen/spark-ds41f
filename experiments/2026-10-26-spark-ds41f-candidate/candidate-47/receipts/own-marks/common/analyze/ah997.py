import json, sys
P = sys.argv[1]
runs = ["1107-boot1", "c47-boot1", "c47-boot2", "1107-boot2"]
d = {r: json.load(open(f"{P}/h997-carveout1-{r}.json")) for r in runs}
for shape in ("tp4-ring", "tp3-triangle"):
    print(shape)
    for r in runs:
        s = d[r]["shapes"][shape]
        print(f"  {r}: cuda_alloc {s['layer_cuda_allocated_MiB']} MiB carveout {s['layer_carveout_MiB']} MiB memavail_drop {s['memavailable_drop_MiB']} MiB mism {s['mismatched_batches']}/{len(s['bits'])*4} rows {s['rows_checked']} tensors>1MiB {s['layer_tensors_MiB']}")
    same = all(d["1107-boot1"]["shapes"][shape]["bits"][m] == d[r]["shapes"][shape]["bits"][m] for r in runs for m in d[r]["shapes"][shape]["bits"])
    print("  cross-image pair hashes identical (all M, 4 batches each):", same)
    for mode in ("eager_us", "graph_us"):
        print("  ", mode, "median(ci95) 1107 b1, b2 | c47 b1, b2 | delta of means | worst c47 - best 1107")
        for m in ["1", "2", "4", "6", "8", "16", "32", "48"]:
            t = {r: d[r]["shapes"][shape]["timing"][m][mode] for r in runs}
            b = [t["1107-boot1"]["mean"], t["1107-boot2"]["mean"]]; c = [t["c47-boot1"]["mean"], t["c47-boot2"]["mean"]]
            hi_c = max(t[r]["mean"] + t[r]["ci95"] for r in ("c47-boot1", "c47-boot2")); lo_b = min(t[r]["mean"] - t[r]["ci95"] for r in ("1107-boot1", "1107-boot2"))
            print(f"    M{m:>3} " + "  ".join(f"{t[r]['mean']:8.2f}±{t[r]['ci95']:.2f}" for r in ("1107-boot1", "1107-boot2", "c47-boot1", "c47-boot2")) + f" | {sum(c)/2-sum(b)/2:+8.2f} | {hi_c-lo_b:+8.2f} | active {d['c47-boot1']['shapes'][shape]['timing'][m]['active_cols']}")
