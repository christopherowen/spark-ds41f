# Hunk-level split of a PR's vLLM patch: keep whole files or whole hunks (0-based, in patch order) and
# drop the rest, without editing any kept line. The kept hunks are written as a patch (new-side line
# numbers recomputed for the dropped hunks), applied on the base (plain, then 3-way), and the result is
# verified to add and remove exactly the kept hunks' lines (as multisets).
# usage: split_hunks.py <N> <label> '<spec json: {"path": "all" | [hunk, ...]}>'
#   -> rebased/<label>.diff (git diff --cached against the base) and splits.json[<label>]
import json, subprocess, os, re, sys, hashlib, collections, glob
W = os.path.expanduser("~/spark-lab/agents/round-55-57-select"); V = f"{W}/c54-vllm"; BASE = "round-55-57-base"
n, label, spec = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
def git(*a, inp=None): return subprocess.run(["git", "-C", V, *a], capture_output=True, text=True, input=inp)
def clean(): git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
src_files = sorted(glob.glob(f"{W}/v/{n}/vllm/*.patch"))
assert len(src_files) == 1, src_files
src = open(src_files[0]).read()
body = src[src.index("\ndiff --git ") + 1:]
body = re.sub(r"(?m)^-- \n[0-9.]+\n*\Z", "", body)
files = re.split(r"(?m)^(?=diff --git )", body)[1:]
out, kept_add, kept_rem, kept = [], collections.Counter(), collections.Counter(), {}
for f in files:
    path = re.match(r"diff --git a/(\S+) b/", f).group(1)
    if path not in spec: continue
    parts = re.split(r"(?m)^(?=@@ )", f)
    header, hunks = parts[0], parts[1:]
    want = range(len(hunks)) if spec[path] == "all" else spec[path]
    shift, text = 0, header
    for i, h in enumerate(hunks):
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", h)
        old_n = int(m.group(2) or 1); new_n = int(m.group(4) or 1)
        if i not in want:
            shift -= new_n - old_n
            continue
        new_start = int(m.group(3)) + shift
        h = re.sub(r"^@@ -(\d+(?:,\d+)?) \+\d+", lambda mm: f"@@ -{mm.group(1)} +{new_start}", h, count=1)
        text += h
        for l in h.splitlines()[1:]:
            if l.startswith("+"): kept_add[l[1:]] += 1
            elif l.startswith("-"): kept_rem[l[1:]] += 1
    kept[path] = "all" if spec[path] == "all" else list(want)
    out.append(text)
missing = sorted(set(spec) - set(kept)); assert not missing, missing
patch = "".join(out)
os.makedirs(f"{W}/splits", exist_ok=True)
open(f"{W}/splits/{label}.patch", "w").write(patch)
clean()
r = git("apply", "--index", f"{W}/splits/{label}.patch"); how = "plain"
if r.returncode:
    clean(); r = git("apply", "-3", "--index", f"{W}/splits/{label}.patch"); how = "3-way"
    if r.returncode:
        conflicts = sorted(set(re.findall(r"^(?:UU|AA) (\S+)", git("status", "--porcelain").stdout, re.M)))
        clean()
        rec = dict(source=os.path.relpath(src_files[0], W), kept=kept, result="conflict", conflicts=conflicts, stderr=r.stderr[-800:])
        print(label, "CONFLICT", conflicts)
        allrec = json.load(open(f"{W}/splits.json")) if os.path.exists(f"{W}/splits.json") else {}
        allrec[label] = rec; json.dump(allrec, open(f"{W}/splits.json", "w"), indent=1); sys.exit(0)
diff = git("diff", "--cached", "--binary", BASE).stdout
def lines(d, sign):
    c = collections.Counter()
    for l in d.splitlines():
        if l.startswith(sign) and not l.startswith(sign * 3): c[l[1:]] += 1
    return c
ok = lines(diff, "+") == kept_add and lines(diff, "-") == kept_rem
open(f"{W}/rebased/{label}.diff", "w").write(diff)
clean()
rec = dict(source=os.path.relpath(src_files[0], W), source_sha256=hashlib.sha256(src.encode()).hexdigest(), kept=kept,
           applied=how, lines_exact=ok, sha256=hashlib.sha256(diff.encode()).hexdigest(),
           files=sorted(set(re.findall(r"^diff --git a/(\S+)", diff, re.M))))
allrec = json.load(open(f"{W}/splits.json")) if os.path.exists(f"{W}/splits.json") else {}
allrec[label] = rec; json.dump(allrec, open(f"{W}/splits.json", "w"), indent=1)
print(label, how, "lines_exact" if ok else "LINES DIFFER", rec["sha256"][:16], len(rec["files"]), "files")
