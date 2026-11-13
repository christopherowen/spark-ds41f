# Run on sp8 in ~/spark-lab/agents/round-51-53-select.
# Apply each pool PR's vLLM patches alone on the base tree (candidate-50 minus 0778, branch round-51-53-base),
# first plainly (git apply --index), then, if that fails, 3-way (git apply -3 --index) to tell context drift
# from a real conflict. Writes alone.json; 3-way-clean results are kept as rebased/<N>.diff for the pair checks.
import json, subprocess, os, re, sys
W = os.path.expanduser("~/spark-lab/agents/round-51-53-select"); V = f"{W}/base-vllm"; BASE = "round-51-53-base"
scan = json.load(open(f"{W}/scan.json"))
def git(*a, inp=None): return subprocess.run(["git", "-C", V, *a], capture_output=True, text=True, input=inp)
def clean(): git("reset", "-q", "--hard", BASE); git("clean", "-qfdx", "-e", "!.git")
def vpatches(n):
    o = scan[str(n)]; ps = [e for e in o["patches"] if e["component"] == "vllm"]
    ex = [e for e in ps if e["path"].startswith("experiments/")]
    ps = ex or ps
    seen = set(); out = []
    for e in sorted(ps, key=lambda e: os.path.basename(e["path"])):
        b = os.path.basename(e["path"])
        if b in seen: continue
        seen.add(b); out.append(e)
    return out
os.makedirs(f"{W}/rebased", exist_ok=True)
res = {}
base_tree = git("rev-parse", f"{BASE}^{{tree}}").stdout.strip()
for n in sorted(int(k) for k in scan):
    ps = vpatches(n)
    if not ps: continue
    r = dict(patches=[os.path.basename(e["path"]) for e in ps], plain="ok", plain_fail=[], three_way=None, conflicts=[], three_way_note="")
    clean()
    for e in ps:
        fn = f"{W}/v/{n}/vllm/{os.path.basename(e['path'])}"
        p = git("apply", "--index", fn)
        if p.returncode:
            r["plain"] = "fail"
            r["plain_fail"] = sorted(set(re.findall(r"patch failed: (\S+?):\d+", p.stderr)) | set(re.findall(r"error: (\S+): (?:does not exist|already exists)", p.stderr))) or p.stderr.strip().splitlines()[:3]
            break
    if r["plain"] == "fail":
        clean(); ok = True
        for e in ps:
            fn = f"{W}/v/{n}/vllm/{os.path.basename(e['path'])}"
            p = git("apply", "-3", "--index", fn)
            if p.returncode:
                ok = False
                r["conflicts"] = sorted(set(re.findall(r"U (\S+)", git("status", "--short").stdout)) | set(re.findall(r"error: (\S+): ", p.stderr)))
                r["three_way_note"] = "\n".join(l for l in p.stderr.splitlines() if "conflict" in l.lower() or l.startswith("error"))[:600]
                break
        r["three_way"] = "clean" if ok else "conflict"
        if ok: open(f"{W}/rebased/{n}.diff", "w").write(git("diff", "--cached", "--binary", BASE).stdout)
    else:
        open(f"{W}/rebased/{n}.diff", "w").write(git("diff", "--cached", "--binary", BASE).stdout)
    res[n] = r
    print(n, r["plain"], r["plain_fail"], r["three_way"] or "", r["conflicts"])
clean()
json.dump(dict(base_branch=BASE, base_commit=git("rev-parse", BASE).stdout.strip(), base_tree=base_tree, results=res), open(f"{W}/alone.json", "w"), indent=1)
