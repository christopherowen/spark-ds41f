# Context-only rebase of a PR patch (or a split of one) whose conflicts with the base are only of two kinds:
#  - test files where both sides append at the same spot: resolved "base, then PR" (the union keeps both);
#  - any file where, against the merge base, the PR only INSERTS lines next to a region the base changed
#    (theirs = X + orig or orig + X): resolved as X placed before (or after) the base's version, so no line
#    of either side is edited.
# Anything else (the PR edits a line the base also edits) is refused as semantic. Applies 3-way with diff3
# conflict markers, resolves, then verifies that the rebased diff adds and removes exactly the source's lines
# (as multisets; blank separator lines may differ). Writes rebased/<label>.diff and rebase-context.json[label].
# usage: rebase_context.py <label> <source patch, relative to the workspace>
import json, subprocess, os, re, sys, collections, hashlib
W = os.path.expanduser("~/spark-lab/agents/round-55-57-select"); V = f"{W}/c54-vllm"; BASE = "round-55-57-base"
label, srcpath = sys.argv[1], sys.argv[2]
def git(*a): return subprocess.run(["git", "-C", V, "-c", "merge.conflictstyle=diff3", *a], capture_output=True, text=True)
def lines(diff, sign):
    out = collections.Counter()
    i = 0 if diff.startswith("diff --git ") else diff.find("\ndiff --git ")
    diff = diff[i:] if i >= 0 else diff
    diff = re.sub(r"(?m)^-- \n[0-9.]+\n*$", "", diff)
    for l in diff.splitlines():
        if l.startswith(sign) and not l.startswith(sign * 3): out[l[1:]] += 1
    return out
CONF = re.compile(r"^<<<<<<< [^\n]*\n(.*?)^\|\|\|\|\|\|\| [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n", re.M | re.S)
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
src = open(f"{W}/{srcpath}").read()
r = git("apply", "-3", "--index", f"{W}/{srcpath}")
unmerged = re.findall(r"^(?:UU|AA) (\S+)", git("status", "--porcelain").stdout, re.M)
kinds, refused = {}, []
for f in unmerged:
    p = f"{V}/{f}"; txt = open(p).read(); fk = []
    def resolve(m):
        ours, orig, theirs = m.group(1), m.group(2), m.group(3)
        if f.startswith("tests/"):
            fk.append("test-append union (base, then PR)")
            return ours.rstrip("\n") + "\n\n\n" + theirs.lstrip("\n")
        if orig and theirs.endswith(orig):
            fk.append("PR inserts before a region the base changed")
            return theirs[: len(theirs) - len(orig)] + ours
        if orig and theirs.startswith(orig):
            fk.append("PR inserts after a region the base changed")
            return ours + theirs[len(orig):]
        if not orig:
            fk.append("both insert at one spot (code): refused")
            refused.append(f); return m.group(0)
        fk.append("PR edits a line the base edits: refused"); refused.append(f); return m.group(0)
    txt = CONF.sub(resolve, txt)
    kinds[f] = fk
    if f not in refused: open(p, "w").write(txt); git("add", f)
if refused:
    rep = dict(result="semantic", refused=sorted(set(refused)), kinds=kinds)
else:
    diff = git("diff", "--cached", "--binary", BASE).stdout
    a_extra = {k: v for k, v in (lines(diff, "+") - lines(src, "+")).items() if k.strip()}
    a_miss = {k: v for k, v in (lines(src, "+") - lines(diff, "+")).items() if k.strip()}
    r_extra = dict(lines(diff, "-") - lines(src, "-")); r_miss = dict(lines(src, "-") - lines(diff, "-"))
    ok = not a_extra and not a_miss and not r_extra and not r_miss
    open(f"{W}/rebased/{label}.diff", "w").write(diff)
    rep = dict(result="context-only" if ok else "NOT-context-only", source=srcpath,
               source_sha256=hashlib.sha256(src.encode()).hexdigest(), resolved=kinds,
               extra_added=a_extra, missing_added=a_miss, extra_removed=r_extra, missing_removed=r_miss,
               blank_lines_delta=lines(diff, "+")[""] - lines(src, "+")[""],
               sha256=hashlib.sha256(diff.encode()).hexdigest())
print(label, rep["result"], json.dumps(kinds))
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
old = json.load(open(f"{W}/rebase-context.json")) if os.path.exists(f"{W}/rebase-context.json") else {}
old[label] = rep; json.dump(old, open(f"{W}/rebase-context.json", "w"), indent=1)
