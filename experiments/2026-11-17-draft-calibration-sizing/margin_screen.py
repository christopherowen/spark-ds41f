"""A screen's arms side by side: ``python margin_screen.py <dir> [<label> ...]`` over
``<dir>/<label>/{c1,c16}.json`` (servebench results) and
``<dir>/greedy-<label>.jsonl`` (the determinism gate's summary line).

For each workload: every arm's throughput, step rate, tokens committed and rows
scheduled a step, drafts accepted a draft step. Without labels, the margin screen's
four arms and margin on against off, as the
ratio of the two boots' means and as the mean of the paired differences of the
workload's batches (the same seed gives the arms the same prompts in the same order;
95% interval, t over the pairs)."""
import json
import math
import os
import sys

OFF, ON = ("margin-off", "margin-off-2"), ("margin-on", "margin-on-2")
KEYS = (("tps", None), ("steps/s", "steps_per_s"), ("tokens/step", "output_tokens_per_step"),
        ("rows/step", "scheduled_tokens_per_step"), ("accepted/draft", "accepted_per_draft"),
        ("drafted/draft", "draft_tokens_per_draft"))
T975 = {9: 2.262, 19: 2.093, 29: 2.045}


def load(root, label, name):
    path = os.path.join(root, label, name + ".json")
    if not os.path.exists(path):
        return None
    s = json.load(open(path))["summary"]
    out = {"failed": s["requests"]["failed"], "tps": s["output_tps"]["mean"], "tps_batches": s["output_tps"]["batches"]}
    for title, key in KEYS[1:]:
        out[title] = s["server"][key]["mean"]
        out[title + "_batches"] = s["server_batches"][key]
    for k in ("survival", "coverage"):
        out[k] = [round(s["server"][f"{k}_at_{i}"]["mean"], 4) for i in range(5)]
    return out


def paired(on, off):
    d = [100 * (a / b - 1) for a, b in zip(on, off)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return [round(mean, 2), round(mean - half, 2), round(mean + half, 2)]


def main(root, labels=()):
    res = {}
    labels = tuple(labels) or OFF + ON
    for name in ("c1", "c16"):
        arms = {label: load(root, label, name) for label in labels}
        arms = {k: v for k, v in arms.items() if v}
        print(f"\n{name}" + "".join(f"{t:>16s}" for t, _ in KEYS) + "   failed")
        for label, a in arms.items():
            print(f"  {label:14s}" + "".join(f"{a[t]:16.4f}" for t, _ in KEYS) + f"   {a['failed']}")
            if name == "c1":
                print(f"  {'':14s}survival by depth {a['survival']}  verified by depth {a['coverage']}")
        res[name] = {"arms": {k: {t: v[t] for t, _ in KEYS} | {"failed": v["failed"], "survival": v["survival"], "coverage": v["coverage"]} for k, v in arms.items()}}
        if all(label in arms for label in OFF + ON):
            cmp = {}
            for t, _ in KEYS:
                on = sum(arms[label][t] for label in ON) / 2
                off = sum(arms[label][t] for label in OFF) / 2
                key = "tps_batches" if t == "tps" else t + "_batches"
                pairs_on = arms[ON[0]][key] + arms[ON[1]][key]
                pairs_off = arms[OFF[0]][key] + arms[OFF[1]][key]
                cmp[t] = {"on": round(on, 4), "off": round(off, 4), "ratio_pct": round(100 * (on / off - 1), 2),
                          "paired_batches_pct": paired(pairs_on, pairs_off),
                          "boots_pct": [round(100 * (arms[ON[i]][t] / arms[OFF[i]][t] - 1), 2) for i in range(2)]}
            res[name]["on_vs_off"] = cmp
            print(f"  {'on against off':14s}" + "".join(f"{cmp[t]['ratio_pct']:+15.2f}%" for t, _ in KEYS))
            print(f"  {'each boot pair':14s}" + "".join(f"{str(cmp[t]['boots_pct']):>16s}" for t, _ in KEYS))
            print(f"  {'paired batches':14s}" + "".join(f"{cmp[t]['paired_batches_pct'][0]:+15.2f}%" for t, _ in KEYS))
            print(f"  {'  95% interval':14s}" + "".join(f"{str(cmp[t]['paired_batches_pct'][1:]):>16s}" for t, _ in KEYS))
    greedy = {}
    for label in labels:
        path = os.path.join(root, f"greedy-{label}.jsonl")
        if os.path.exists(path):
            lines = [json.loads(line) for line in open(path) if line.strip().startswith("{")]
            summary = [x for x in lines if x.get("summary")][-1]
            greedy[label] = {"identical": summary.get("identical"),
                             "token_diffs": sum(v.get("token_diffs", 0) for v in summary.get("scenarios", {}).values()),
                             "reference_sha256": str(summary.get("reference_sha256", ""))[:8],
                             "long_sha256": str(summary.get("long_sha256", ""))[:8]}
    res["greedy"] = greedy
    print("\ngreedy records")
    for label, g in greedy.items():
        print(f"  {label:14s} {g}")
    json.dump(res, open(os.path.join(root, "margin-screen.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
