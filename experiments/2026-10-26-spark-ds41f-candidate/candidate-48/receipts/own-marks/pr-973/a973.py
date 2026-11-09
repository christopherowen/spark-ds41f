"""Summarize PR #973 bench JSONs: per scenario and stage, both boots of 1107 and candidate-48,
the 2-boot mean +/- half-range, speedup, whether the gain is beyond the interval, and tokens_sha256 equality."""
import json
import sys

d = sys.argv[1]
runs = {}
for arm in ("baseline1107", "candidate48"):
    for b in (1, 2):
        runs[(arm, b)] = json.load(open(f"{d}/bench-{arm}-boot{b}.json"))

stages = ("parse_chat_messages", "render_only", "cold_e2e", "incremental_e2e")
out = []
n = len(runs[("baseline1107", 1)]["scenarios"])
all_sha_equal = True
all_gain_beyond = True
for i in range(n):
    sc = {k: v["scenarios"][i] for k, v in runs.items()}
    s0 = sc[("baseline1107", 1)]
    key = {k: s0[k] for k in s0 if k in ("n_turns", "user_every", "include_control_spelling", "num_messages", "num_tokens")}
    shas = {f"{a}-boot{b}": s["tokens_sha256"] for (a, b), s in sc.items()}
    sha_equal = len(set(shas.values())) == 1
    all_sha_equal &= sha_equal
    row = {"scenario": key, "tokens_sha256": sorted(set(shas.values())), "tokens_sha256_equal_all_4_runs": sha_equal, "stages": {}}
    for st in stages:
        b = [sc[("baseline1107", k)][st]["mean_ms"] for k in (1, 2)]
        c = [sc[("candidate48", k)][st]["mean_ms"] for k in (1, 2)]
        bm, bh = sum(b) / 2, abs(b[0] - b[1]) / 2
        cm, ch = sum(c) / 2, abs(c[0] - c[1]) / 2
        beyond = (cm + ch) < (bm - bh)
        if st in ("cold_e2e", "incremental_e2e"):
            all_gain_beyond &= beyond
        row["stages"][st] = {
            "baseline1107_mean_ms_boot1_boot2": b,
            "candidate48_mean_ms_boot1_boot2": c,
            "baseline1107_ms": f"{bm:.3f} +/- {bh:.3f}",
            "candidate48_ms": f"{cm:.3f} +/- {ch:.3f}",
            "speedup_x": round(bm / cm, 2) if cm else None,
            "gain_beyond_interval": beyond,
        }
    row["peak_heap_kib"] = {f"{a}-boot{b}": s.get("peak_heap_kib") for (a, b), s in sc.items()}
    out.append(row)
print(json.dumps({"all_tokens_sha256_equal": all_sha_equal, "all_e2e_gains_beyond_interval": all_gain_beyond, "scenarios": out}, indent=1))
