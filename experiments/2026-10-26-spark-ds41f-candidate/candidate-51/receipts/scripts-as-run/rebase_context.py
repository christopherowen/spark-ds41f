# Context-only rebase of a PR patch whose only conflict with the base is in test files where both sides
# append at the same spot: apply 3-way, resolve each test-file conflict as "base, then PR" (the union keeps
# both), refuse if any non-test file conflicts, and verify that the rebased diff adds and removes exactly the
# PR's lines (as multisets). Writes rebased/<N>.diff and rebase-<N>.json.
import json, subprocess, os, re, sys, collections, hashlib
W = os.path.expanduser("~/spark-lab/agents/round-51-53-select"); V = f"{W}/base-vllm"; BASE = "round-51-53-base"
alone = json.load(open(f"{W}/alone.json"))["results"]
def git(*a): return subprocess.run(["git", "-C", V, *a], capture_output=True, text=True)
def lines(diff, sign):
    out = collections.Counter()
    i = 0 if diff.startswith("diff --git ") else diff.find("\ndiff --git ")
    diff = diff[i:] if i >= 0 else diff
    diff = re.sub(r"(?m)^-- \n[0-9.]+\n*$", "", diff)  # format-patch signature
    for l in diff.splitlines():
        if l.startswith(sign) and not l.startswith(sign * 3): out[l[1:]] += 1
    return out
rep = {}
for n in sys.argv[1:]:
    r = alone[n]; git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
    src = ""
    for p in r["patches"]:
        fn = f"{W}/v/{n}/vllm/{p}"; src += open(fn).read(); git("apply", "-3", "--index", fn)
    status = git("status", "--porcelain").stdout
    unmerged = re.findall(r"^(?:UU|AA) (\S+)", status, re.M)
    bad = [f for f in unmerged if not f.startswith("tests/")]
    if bad: rep[n] = dict(result="code-conflict", files=bad); print(n, "code conflict", bad); continue
    for f in unmerged:
        p = f"{V}/{f}"; txt = open(p).read()
        txt = re.sub(r"^<<<<<<< .*?\n(.*?)^=======\n(.*?)^>>>>>>> .*?\n", lambda m: m.group(1).rstrip("\n") + "\n\n\n" + m.group(2).lstrip("\n"), txt, flags=re.M | re.S)
        open(p, "w").write(txt); git("add", f)
    diff = git("diff", "--cached", "--binary", BASE).stdout
    a_ok = lines(diff, "+") - lines(src, "+"); a_miss = lines(src, "+") - lines(diff, "+")
    r_ok = lines(diff, "-") - lines(src, "-"); r_miss = lines(src, "-") - lines(diff, "-")
    extra_added = {k: v for k, v in a_ok.items() if k.strip()}  # blank separator lines may differ
    a_miss = collections.Counter({k: v for k, v in a_miss.items() if k.strip()})
    ok = not extra_added and not a_miss and not r_ok and not r_miss
    open(f"{W}/rebased/{n}.diff", "w").write(diff)
    rep[n] = dict(result="context-only" if ok else "NOT-context-only", resolved_test_files=unmerged,
                  extra_added=dict(extra_added), missing_added=dict(a_miss), extra_removed=dict(r_ok), missing_removed=dict(r_miss),
                  blank_lines_delta=lines(diff, "+")[""] - lines(src, "+")[""],
                  sha256=hashlib.sha256(diff.encode()).hexdigest())
    print(n, rep[n]["result"], unmerged, "blank delta", rep[n]["blank_lines_delta"])
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
old = json.load(open(f"{W}/rebase-context.json")) if os.path.exists(f"{W}/rebase-context.json") else {}
old.update(rep); json.dump(old, open(f"{W}/rebase-context.json", "w"), indent=1)
