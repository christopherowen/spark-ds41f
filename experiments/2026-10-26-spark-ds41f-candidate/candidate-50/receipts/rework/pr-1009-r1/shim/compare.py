"""Compare attention-harness runs across arms and boots: errors, and differing bytes of
layer 24's attention output (WO's input), top-k indices and SWA cache pages, per TP
size and step. Every rank's digest is compared; rank 0's tensors byte for byte.

usage: python3 compare.py <json dir> <out.json>
"""

import json
import sys
from pathlib import Path

import torch

d = Path(sys.argv[1])
ARMS = ("c50", "fix", "no1009")
WORLDS = (3, 4, 8)
BOOTS = (1, 2, 3)  # 3: the final tree's fix arm (docstring as shipped)


def load(arm, world, boot):
    j = d / f"{arm}-tp{world}-boot{boot}.json"
    p = d / f"{arm}-tp{world}-boot{boot}.pt"
    if not j.exists():
        return None
    return json.loads(j.read_text()), torch.load(p, weights_only=False)


def ndiff(a, b):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def pair(x, y):
    """Differences between two runs' steps (x, y: (json, pt))."""
    out = {}
    jx = {s["scenario"]: s for s in x[0]["scenarios"]}
    jy = {s["scenario"]: s for s in y[0]["scenarios"]}
    for name in jx:
        a, b = jx[name], jy.get(name)
        if b is None:
            continue
        e = {}
        if a["error"] or b["error"]:
            e["errors"] = [a["error"], b["error"]]
        e["out20_bytes"] = ndiff(x[1][name]["out20"], y[1][name]["out20"])
        if not (a["error"] or b["error"]):
            ra, rb = x[1][name]["rank0"], y[1][name]["rank0"]
            e["rank0_bytes"] = {k: ndiff(ra[k], rb[k]) for k in ("out", "topk", "swa")}
            e["ranks_digest_equal"] = all(
                all(p[k] == q[k] for k in ("out", "topk", "swa"))
                for p, q in zip(a["ranks"], b["ranks"])
            ) and len(a["ranks"]) == len(b["ranks"])
            e["bit_exact"] = e["ranks_digest_equal"] and all(
                v == 0 for v in e["rank0_bytes"].values()
            ) and e["out20_bytes"] == 0
        out[name] = e
    return out


report = {"runs": {}, "pairs": {}}
for world in WORLDS:
    runs = {(arm, boot): load(arm, world, boot) for arm in ARMS for boot in BOOTS}
    for (arm, boot), r in runs.items():
        if r is None:
            continue
        report["runs"][f"{arm}-tp{world}-boot{boot}"] = [
            dict(scenario=s["scenario"], T=s["T"], is_decode=s["is_decode"],
                 num_decode_tokens=s["num_decode_tokens"], error=s["error"],
                 error_rank=s.get("error_rank"), passes=s["passes"], converged=s["converged"],
                 ranks_agree_on_output=s.get("ranks_agree_on_output"),
                 local_rows=[x["local_rows"] for x in s["ranks"]],
                 real_rows=[x["real_rows"] for x in s["ranks"]],
                 gathers=[x["gathers"] for x in s["ranks"]][:1])
            for s in r[0]["scenarios"]
        ]
    for name, (ka, kb) in {
        "fix_vs_no1009": (("fix", 1), ("no1009", 1)),
        "fix_vs_no1009_boot2": (("fix", 2), ("no1009", 2)),
        "fix_vs_c50": (("fix", 1), ("c50", 1)),
        "c50_vs_no1009": (("c50", 1), ("no1009", 1)),
        "fix_boot1_vs_boot2": (("fix", 1), ("fix", 2)),
        "no1009_boot1_vs_boot2": (("no1009", 1), ("no1009", 2)),
        "fix_final_vs_no1009": (("fix", 3), ("no1009", 1)),
        "fix_final_vs_fix_boot1": (("fix", 3), ("fix", 1)),
    }.items():
        if runs.get(ka) is None or runs.get(kb) is None:
            continue
        report["pairs"][f"tp{world}:{name}"] = pair(runs[ka], runs[kb])

Path(sys.argv[2]).write_text(json.dumps(report, indent=1) + "\n")
for k, v in report["pairs"].items():
    for s, e in v.items():
        print(k, s, "bit_exact" if e.get("bit_exact") else e)
