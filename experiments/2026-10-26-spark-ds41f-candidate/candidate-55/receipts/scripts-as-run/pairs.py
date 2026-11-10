# For every pair of patch sets (rebased/<label>.diff: the PR's patch as it applies on the base, plain or
# context-only rebased, or a file-level split) that touch a common file: apply A (git apply --index),
# then check B (git apply --check); both orders. Writes pairs.json.
import json, subprocess, os, re, itertools, sys
W = os.path.expanduser("~/spark-lab/agents/round-55-57-select"); V = f"{W}/c54-vllm"; BASE = "round-55-57-base"
labels = sys.argv[1:]
def git(*a): return subprocess.run(["git", "-C", V, *a], capture_output=True, text=True)
def files(l): return set(re.findall(r"^diff --git a/(\S+)", open(f"{W}/rebased/{l}.diff").read(), re.M))
F = {l: files(l) for l in labels}
pairs = [(a, b) for a, b in itertools.combinations(labels, 2) if F[a] & F[b]]
out = []
for a, b in pairs:
    rec = dict(a=a, b=b, common=sorted(F[a] & F[b]))
    for x, y in ((a, b), (b, a)):
        git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
        r = git("apply", "--index", f"{W}/rebased/{x}.diff"); assert r.returncode == 0, (x, r.stderr)
        c = git("apply", "--check", f"{W}/rebased/{y}.diff")
        rec[f"{y}_after_{x}"] = "clean" if c.returncode == 0 else sorted(set(re.findall(r"patch failed: (\S+?):\d+", c.stderr))) or c.stderr.strip().splitlines()[:2]
    rec["conflict"] = any(v != "clean" for k, v in rec.items() if "_after_" in k)
    out.append(rec)
    if rec["conflict"]: print("CONFLICT", a, b, {k: v for k, v in rec.items() if "_after_" in k})
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
json.dump(dict(labels=labels, overlapping_pairs=len(pairs), conflicting_pairs=sum(r["conflict"] for r in out), pairs=out), open(f"{W}/pairs.json", "w"), indent=1)
print(len(labels), "sets;", len(pairs), "overlapping pairs;", sum(r["conflict"] for r in out), "conflict")
