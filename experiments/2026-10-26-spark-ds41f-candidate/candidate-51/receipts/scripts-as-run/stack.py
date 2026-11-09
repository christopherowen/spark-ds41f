# Stack each proposed candidate on the base in order, on sp8: a PR patch that applies as published goes in
# with git am; a context-only rebase or a file-level split goes in as its diff, committed with the PR
# commit's author, date and message (a split adds one line saying which files it keeps). Every patch is
# exported with format-patch to out/<cand>/; stack.json records order, method, SHA-256 and the stack tree.
import json, subprocess, os, re, sys, hashlib
W = os.path.expanduser("~/spark-lab/agents/round-51-53-select"); V = f"{W}/base-vllm"; BASE = "round-51-53-base"
alone = json.load(open(f"{W}/alone.json"))["results"]; splits = json.load(open(f"{W}/splits.json"))
C = json.loads(sys.argv[1])
env = dict(os.environ, GIT_COMMITTER_NAME="b", GIT_COMMITTER_EMAIL="b@b", GIT_COMMITTER_DATE="2026-01-01T00:00:00Z")
def git(*a, inp=None): return subprocess.run(["git", "-C", V, "-c", "user.name=b", "-c", "user.email=b@b", *a], capture_output=True, text=True, input=inp, env=env)
def info(patch):
    msg = subprocess.run(["git", "mailinfo", "/dev/stdout", "/dev/null"], input=open(patch).read(), capture_output=True, text=True)
    hdr, body = {}, ""
    meta = msg.stdout
    for l in meta.splitlines():
        if ": " in l: k, v = l.split(": ", 1); hdr[k] = v
    full = subprocess.run(["git", "mailinfo", "/tmp/msg.txt", "/dev/null"], input=open(patch).read(), capture_output=True, text=True)
    body = open("/tmp/msg.txt").read()
    return hdr, body
res = {}
for cand, labels in C.items():
    git("am", "--abort"); git("checkout", "-q", "-f", BASE); git("clean", "-qfd")
    git("checkout", "-q", "-B", f"cand-{cand}", BASE)
    out = f"{W}/out/{cand}"; os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(f"{out}/{f}")
    rows = []
    for i, lab in enumerate(labels):
        pr = lab.split("-")[0]
        orig = f"{W}/v/{pr}/vllm/{alone[pr]['patches'][0]}"
        assert len(alone[pr]["patches"]) == 1
        if lab in splits or alone[pr]["plain"] != "ok":
            how = "split" if lab in splits else "context-only rebase"
            r = git("apply", "--index", f"{W}/rebased/{lab}.diff")
            if r.returncode:
                r = git("apply", "-3", "--index", f"{W}/rebased/{lab}.diff"); how += " (3-way on stack)"
                if r.returncode: print("FAIL", cand, lab, r.stderr[-400:]); sys.exit(1)
            hdr, body = info(orig)
            msg = hdr["Subject"] + "\n\n" + body.rstrip() + "\n"
            if lab in splits:
                msg += "\nSplit at selection for candidates 51-53: this patch keeps " + ", ".join(splits[lab]["kept_files"]) + " of the PR's patch; every kept line is unchanged.\n"
            r = git("commit", "-q", "--no-verify", "-m", msg, "--author", f"{hdr['Author']} <{hdr['Email']}>", "--date", hdr["Date"])
            assert r.returncode == 0, r.stderr
        else:
            how = "git am"
            r = git("am", "-q", "--keep-cr", orig)
            if r.returncode:
                git("am", "--abort"); r = git("am", "-3", "-q", "--keep-cr", orig); how = "git am -3"
                if r.returncode: print("FAIL", cand, lab, r.stdout[-300:], r.stderr[-300:]); git("am", "--abort"); sys.exit(1)
        data = git("format-patch", "-1", "--stdout", "HEAD").stdout if how != "git am" else open(orig).read()
        name = f"{i + 1:02d}-" + re.sub(r"^\d{4}-", "", os.path.basename(orig))
        open(f"{out}/{name}", "w").write(data)
        rows.append(dict(label=lab, pr=int(pr), source_patch=os.path.basename(orig), method=how, file=name,
                         sha256=hashlib.sha256(data.encode()).hexdigest(), source_sha256=hashlib.sha256(open(orig).read().encode()).hexdigest()))
        print(cand, lab, how)
    res[cand] = dict(order=labels, patches=rows, stack_head=git("rev-parse", "HEAD").stdout.strip(), stack_tree=git("rev-parse", "HEAD^{tree}").stdout.strip())
    print(cand, "tree", res[cand]["stack_tree"])
# The round's union: every candidate's stack on one tree (the candidates overlap in no file).
git("checkout", "-q", "-B", "union", BASE); ok = True
for cand in C:
    for row in res[cand]["patches"]:
        r = git("am", "-q", "--keep-cr", f"{W}/out/{cand}/{row['file']}")
        if r.returncode: ok = False; print("UNION FAIL", cand, row["label"]); git("am", "--abort"); break
res["union"] = dict(applies=ok, tree=git("rev-parse", "HEAD^{tree}").stdout.strip())
git("checkout", "-q", BASE)
json.dump(dict(base_commit=git("rev-parse", BASE).stdout.strip(), base_tree=git("rev-parse", BASE + "^{tree}").stdout.strip(), candidates=res), open(f"{W}/stack.json", "w"), indent=1)
print("union", res["union"])
