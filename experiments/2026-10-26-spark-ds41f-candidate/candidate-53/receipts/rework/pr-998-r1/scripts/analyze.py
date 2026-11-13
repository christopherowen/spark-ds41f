"""Cells for #998 round 1 from the serving-path ABBA (base, fix, fix, base) and the
PR's own script on the fix arm. usage: analyze.py JSON_DIR PR_OUT_DIR > cells.json"""
import json
import re
import sys
from pathlib import Path

jdir, prdir = Path(sys.argv[1]), Path(sys.argv[2])
runs = {name: json.loads((jdir / f"bench-serving-{name}.json").read_text())
        for name in ("base-boot1", "fix-boot1", "fix-boot2", "base-boot2")}
base = [runs["base-boot1"], runs["base-boot2"]]
fix = [runs["fix-boot1"], runs["fix-boot2"]]

cells = {"order": ["base-boot1", "fix-boot1", "fix-boot2", "base-boot2"]}

# Bit-exactness: every output hash equal across all four runs.
keys = sorted(base[0]["hashes"])
mismatch = {k: {n: r["hashes"].get(k) for n, r in runs.items()}
            for k in keys if len({r["hashes"].get(k) for r in runs.values()}) != 1}
outputs = [k for k in keys if not k.startswith(("embed_tokens.", "lm_head.weight", "lm_head.screen",
                                                "markov_w1", "layer"))]
cells["bit_exact"] = {"all_hashes_equal": not mismatch, "compared": len(keys),
                      "output_keys": outputs, "mismatch": mismatch}

mem_keys = ["cuda_allocated_mib", "cuda_reserved_mib", "cuda_peak_reserved_mib",
            "host_mem_available_used_mib", "cuda_free_used_mib",
            "carveout_weights_regular_memory_mib", "carveout_total_mib"]
memory = {}
for k in mem_keys:
    b = [r["memory"][k] for r in base]
    f = [r["memory"][k] for r in fix]
    memory[k] = {"base": b, "fix": f,
                 "saved_mean": round(sum(b) / 2 - sum(f) / 2, 2),
                 "saved_pairs": [round(b[0] - f[0], 2), round(b[1] - f[1], 2)]}
memory["carveout_buffers_mib"] = {"base": [r["memory"]["carveout_buffers_mib"] for r in base],
                                  "fix": [r["memory"]["carveout_buffers_mib"] for r in fix]}
memory["placement_fix"] = fix[0]["memory"]["placement_in_carveout"]
memory["placement_base"] = base[0]["memory"]["placement_in_carveout"]
cells["memory"] = memory

times = {}
for k in base[0]["times_us"]:
    b = [r["times_us"][k] for r in base]
    f = [r["times_us"][k] for r in fix]
    bm, fm = sum(b) / 2, sum(f) / 2
    if max(f) < min(b):
        verdict = "faster beyond interval"
    elif min(f) > max(b):
        verdict = "slower beyond interval"
    else:
        verdict = "within interval"
    times[k] = {"base_us": b, "fix_us": f, "ratio_fix_over_base": round(fm / bm, 4),
                "delta_pct": round((fm / bm - 1) * 100, 1), "verdict": verdict}
cells["times_us"] = times

pr = {}
for boot in (1, 2):
    path = prdir / f"bench-pr998-script-fix-boot{boot}.txt"
    if not path.exists():
        continue
    text = path.read_text()
    start = text.find('{\n  "boot"')
    if start < 0:
        pr[f"boot{boot}"] = {"error": text[-2000:]}
        continue
    data = json.loads(text[start:text.rindex("}") + 1])
    pr[f"boot{boot}"] = {
        shape: {"savings": data[shape]["savings"],
                "carveout_buffers_mib": data[shape]["candidate"]["carveout_buffers_mib"],
                "kernels": {k: {kk: v[kk] for kk in ("regular_us", "carveout_us", "ratio", "bit_identical")}
                            for k, v in data[shape]["kernels"].items()}}
        for shape in ("tp3", "tp4")}
cells["pr_script_on_fix"] = pr
print(json.dumps(cells, indent=1))
