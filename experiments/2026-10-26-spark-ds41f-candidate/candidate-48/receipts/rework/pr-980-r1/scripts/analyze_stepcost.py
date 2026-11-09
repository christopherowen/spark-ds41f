"""analyze_stepcost.py <out-dir>: summarize stepcost-<arm>-<boot>.json.

Per component and grid shape, each arm's per-call wall time (median over rounds)
in both boots; arm value = mean of the two boots, interval = half their range.
Deltas against the 1107 arm, with the two arms' half-ranges summed as the
interval. Geometric means over the grid's 14 decode shapes."""
import json
import math
import os
import sys

out_dir = sys.argv[1]
ARMS = ("1107", "c48", "r1")
BOOTS = ("boot1", "boot2")
data = {}
for arm in ARMS:
    for boot in BOOTS:
        p = os.path.join(out_dir, f"stepcost-{arm}-{boot}.json")
        d = json.load(open(p))
        assert not d["errors"], (p, d["errors"])
        for r in d["results"]:
            key = (r["component"], r["num_reqs"], r["depth"])
            data.setdefault(key, {}).setdefault(arm, {})[boot] = r
        data.setdefault("_meta", {})[(arm, boot)] = {
            k: d[k] for k in ("files_sha256_16", "sampler_use_flashinfer",
                              "broadcast_calls", "est_refits", "utc")
        }

components = sorted({k[0] for k in data if k != "_meta"},
                    key=lambda c: ["greedy_check", "est_step", "est_calibrate",
                                   "record_conf", "get_num_tokens", "greedy_total",
                                   "sample_greedy", "sample_sampled", "draft_predict",
                                   "draft_vp_predict"].index(c))
shapes = sorted({(k[1], k[2]) for k in data if k != "_meta"}, key=lambda s: (s[1], s[0]))

summary = {"metric": "per-call wall time, us (median of 9 rounds x 300 calls)",
           "shapes": [f"{r}x{1 + d}" for r, d in shapes], "components": {}}
for comp in components:
    rows = []
    logs = {arm: [] for arm in ARMS}
    host_logs = {arm: [] for arm in ARMS}
    for r, dpt in shapes:
        cell = data[(comp, r, dpt)]
        row = {"num_reqs": r, "rows_per_req": 1 + dpt}
        for arm in ARMS:
            w = [cell[arm][b]["wall_us_median"] for b in BOOTS]
            h = [cell[arm][b]["host_us_median"] for b in BOOTS]
            row[arm] = {"boot1": w[0], "boot2": w[1],
                        "mean": round(sum(w) / 2, 3), "half_range": round(abs(w[0] - w[1]) / 2, 3),
                        "host_mean": round(sum(h) / 2, 3)}
            logs[arm].append(math.log(sum(w) / 2))
            host_logs[arm].append(math.log(sum(h) / 2))
        for arm in ("c48", "r1"):
            d = row[arm]["mean"] - row["1107"]["mean"]
            row[f"{arm}_minus_1107_us"] = round(d, 3)
            row[f"{arm}_minus_1107_interval_us"] = round(
                row[arm]["half_range"] + row["1107"]["half_range"], 3)
        rows.append(row)
    gm = {arm: math.exp(sum(v) / len(v)) for arm, v in logs.items()}
    hgm = {arm: math.exp(sum(v) / len(v)) for arm, v in host_logs.items()}
    # boot-level geomeans, for a boot-to-boot interval on the geomean ratio
    boot_gm = {}
    for arm in ARMS:
        for b in BOOTS:
            v = [math.log(data[(comp, r, dpt)][arm][b]["wall_us_median"]) for r, dpt in shapes]
            boot_gm[f"{arm}-{b}"] = round(math.exp(sum(v) / len(v)), 3)
    summary["components"][comp] = {
        "geomean_wall_us": {a: round(v, 3) for a, v in gm.items()},
        "geomean_host_us": {a: round(v, 3) for a, v in hgm.items()},
        "geomean_wall_us_by_boot": boot_gm,
        "c48_vs_1107_pct": round(100 * (gm["c48"] / gm["1107"] - 1), 3),
        "r1_vs_1107_pct": round(100 * (gm["r1"] / gm["1107"] - 1), 3),
        "aa_spread_pct": {a: round(100 * abs(boot_gm[f"{a}-boot1"] / boot_gm[f"{a}-boot2"] - 1), 3)
                          for a in ARMS},
        "rows": rows,
    }
summary["meta"] = {f"{a}-{b}": v for (a, b), v in data["_meta"].items()}
print(json.dumps(summary, indent=1))
