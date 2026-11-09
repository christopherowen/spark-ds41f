# File-level split of a PR's (rebased) diff: keep only the named files; every kept line unchanged.
# usage: derive_split.py <N> <label> <file> [<file> ...]  -> rebased/<label>.diff
import subprocess, os, sys, hashlib, json
W = os.path.expanduser("~/spark-lab/agents/round-51-53-select"); V = f"{W}/base-vllm"; BASE = "round-51-53-base"
n, label, keep = sys.argv[1], sys.argv[2], sys.argv[3:]
def git(*a): return subprocess.run(["git", "-C", V, *a], capture_output=True, text=True)
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
r = git("apply", "--index", *[f"--include={f}" for f in keep], f"{W}/rebased/{n}.diff")
assert r.returncode == 0, r.stderr
diff = git("diff", "--cached", "--binary", BASE).stdout
got = sorted(set(l.split()[2][2:] for l in diff.splitlines() if l.startswith("diff --git ")))
assert got == sorted(keep), (got, keep)
open(f"{W}/rebased/{label}.diff", "w").write(diff)
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
rec = json.load(open(f"{W}/splits.json")) if os.path.exists(f"{W}/splits.json") else {}
rec[label] = dict(source=f"rebased/{n}.diff", kept_files=keep, sha256=hashlib.sha256(diff.encode()).hexdigest())
json.dump(rec, open(f"{W}/splits.json", "w"), indent=1)
print(label, rec[label]["sha256"][:16], got)
