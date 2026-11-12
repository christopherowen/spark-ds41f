"""c1 decode of two c1 checks, paired request by request.

Usage: python3 c1_paired_decode.py REFERENCE_RUN_DIR CANDIDATE_RUN_DIR [N]

Each RUN_DIR holds <label>-bootK/c1.json (results/private/boot-differences/<run>/). The first N
requests (default: the fewest any boot sent) must have the same per-step accept decisions on
both sides (c1_decisions.py), so each pair decodes the same tokens. Prints each boot's decode
tok/s over those requests ((completion_tokens - 1) / (end - first), summed) and the geometric
mean change of each request's decode time, its mean over each side's boots, with a 95% bound.
"""
import glob
import json
import math
import statistics as st
import sys


def boots(run_dir):
    out = {}
    for path in sorted(glob.glob(f"{run_dir}/*/c1.json")):
        requests = json.load(open(path))["requests"]
        out[path.split("/")[-2]] = [(r["end"] - r["first"], r["completion_tokens"], r["first"] - r["sent"])
                                    for r in requests]
    return out


reference, candidate = boots(sys.argv[1]), boots(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else min(len(v) for v in [*reference.values(), *candidate.values()])


def rate(requests):
    return sum(tokens - 1 for _, tokens, _ in requests[:n]) / sum(decode for decode, _, _ in requests[:n])


for name, side in (("reference", reference), ("candidate", candidate)):
    rates = [rate(v) for v in side.values()]
    ttft = st.mean(st.mean(v[i][2] for v in side.values()) for i in range(n)) * 1000
    print(f"{name}: decode tok/s per boot over {n} requests {[round(x, 2) for x in rates]}, "
          f"mean {st.mean(rates):.2f}; mean TTFT {ttft:.1f} ms")

ref_time = [st.mean(v[i][0] for v in reference.values()) for i in range(n)]
cand_time = [st.mean(v[i][0] for v in candidate.values()) for i in range(n)]
logs = [math.log(c / r) for r, c in zip(ref_time, cand_time)]
bound = 1.96 * st.stdev(logs) / math.sqrt(n) * 100
print(f"decode time per request, paired: {(math.exp(st.mean(logs)) - 1) * 100:+.2f}% +/- {bound:.2f}%")
mean_rate = lambda side: st.mean(rate(v) for v in side.values())
print(f"decode tok/s change: {(mean_rate(candidate) / mean_rate(reference) - 1) * 100:+.2f}%")
