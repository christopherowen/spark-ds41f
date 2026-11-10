# Stack each proposed candidate on the base in order, on sp8: a PR patch that applies as published goes in
# with git am; a context-only rebase or a hunk-level split goes in as its diff, committed with the PR
# commit's author, date and message (a split adds a paragraph naming the files and hunks it keeps). Every
# patch is exported with format-patch to out/<cand>/; stack.json records order, method, SHA-256 and trees.
# Each candidate is built on its own branch from the base (git checkout -B, never a reset of another branch).
import json, subprocess, os, re, sys, hashlib, glob
W = os.path.expanduser("~/spark-lab/agents/round-55-57-select"); V = f"{W}/c54-vllm"; BASE = "round-55-57-base"
alone = json.load(open(f"{W}/alone.json"))["results"]; splits = json.load(open(f"{W}/splits.json"))
ctx = {k: v for k, v in json.load(open(f"{W}/rebase-context.json")).items() if v["result"] == "context-only"}
C = json.loads(sys.argv[1])
env = dict(os.environ, GIT_COMMITTER_NAME="b", GIT_COMMITTER_EMAIL="b@b", GIT_COMMITTER_DATE="2026-01-01T00:00:00Z")
def git(*a, inp=None): return subprocess.run(["git", "-C", V, "-c", "user.name=b", "-c", "user.email=b@b", *a], capture_output=True, text=True, input=inp, env=env)
def info(patch):
    tmp = f"{W}/tmp-msg.txt"
    meta = subprocess.run(["git", "mailinfo", tmp, "/dev/null"], input=open(patch).read(), capture_output=True, text=True).stdout
    hdr = dict(l.split(": ", 1) for l in meta.splitlines() if ": " in l)
    return hdr, open(tmp).read()
res = {}
for cand, labels in C.items():
    git("am", "--abort"); git("checkout", "-q", "-f", BASE); git("clean", "-qfd")
    git("checkout", "-q", "-B", f"cand-{cand}", BASE)
    out = f"{W}/out/{cand}"; os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(f"{out}/{f}")
    rows = []
    for i, lab in enumerate(labels):
        pr = lab.split("-")[0]
        srcs = sorted(glob.glob(f"{W}/v/{pr}/vllm/*.patch")); assert len(srcs) == 1; orig = srcs[0]
        if lab in splits or lab in ctx:
            how = "hunk-level split" if lab in splits else "context-only rebase"
            r = git("apply", "--index", f"{W}/rebased/{lab}.diff")
            if r.returncode: print("FAIL", cand, lab, r.stderr[-400:]); sys.exit(1)
            hdr, body = info(orig)
            msg = hdr["Subject"] + "\n\n" + body.rstrip() + "\n"
            if lab in splits:
                kept = splits[lab]["kept"]
                desc = "; ".join(f + (" (whole)" if v == "all" else " (hunks " + ", ".join(str(h + 1) for h in v) + " of the PR's)") for f, v in kept.items())
                msg += ("\nSplit at selection for candidates 55-57 (" + lab + "): this patch keeps " + desc +
                        ". Every kept line is unchanged; the other hunks are left out (receipts/selection-proposal.json).\n")
            r = git("commit", "-q", "--no-verify", "-m", msg, "--author", f"{hdr['Author']} <{hdr['Email']}>", "--date", hdr["Date"])
            assert r.returncode == 0, r.stderr
        else:
            assert alone[pr]["plain"] == "ok", lab
            how = "git am"
            r = git("am", "-q", "--keep-cr", orig)
            if r.returncode: print("FAIL", cand, lab, r.stdout[-300:], r.stderr[-300:]); git("am", "--abort"); sys.exit(1)
        data = git("format-patch", "-1", "--stdout", "HEAD").stdout if how != "git am" else open(orig).read()
        name = f"{i + 1:02d}-" + (f"{lab}-" if lab in splits else "") + re.sub(r"^\d{4}-", "", os.path.basename(orig))
        open(f"{out}/{name}", "w").write(data)
        rows.append(dict(label=lab, pr=int(pr), source_patch=os.path.basename(orig), method=how, file=name,
                         sha256=hashlib.sha256(data.encode()).hexdigest(),
                         source_sha256=hashlib.sha256(open(orig).read().encode()).hexdigest()))
        print(cand, lab, how)
    res[cand] = dict(order=labels, patches=rows, stack_head=git("rev-parse", "HEAD").stdout.strip(),
                     stack_tree=git("rev-parse", "HEAD^{tree}").stdout.strip(),
                     check=git("diff", "--check", BASE, "HEAD").stdout.strip() or "clean")
    print(cand, "tree", res[cand]["stack_tree"], "diff --check", res[cand]["check"][:200])
# The round's union: every candidate's stack on one tree.
git("checkout", "-q", "-B", "union", BASE); ok = True
for cand in C:
    for row in res[cand]["patches"]:
        r = git("am", "-q", "--keep-cr", f"{W}/out/{cand}/{row['file']}")
        if r.returncode: ok = False; print("UNION FAIL", cand, row["label"]); git("am", "--abort"); break
res["union"] = dict(applies=ok, tree=git("rev-parse", "HEAD^{tree}").stdout.strip())
git("checkout", "-q", BASE)
json.dump(dict(base_commit=git("rev-parse", BASE).stdout.strip(), base_tree=git("rev-parse", BASE + "^{tree}").stdout.strip(), candidates=res), open(f"{W}/stack.json", "w"), indent=1)
print("union", res["union"])
