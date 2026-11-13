# For PRs whose patch conflicts on the base, apply 3-way and print each conflict region (ours = base, theirs = PR),
# truncated, to tell context-only (test appends at the same spot) from code conflicts.
import json, subprocess, os, re, sys
W = os.path.expanduser("~/spark-lab/agents/round-51-53-select"); V = f"{W}/base-vllm"; BASE = "round-51-53-base"
alone = json.load(open(f"{W}/alone.json"))["results"]
def git(*a): return subprocess.run(["git", "-C", V, *a], capture_output=True, text=True)
out = {}
for n in sys.argv[1:]:
    r = alone[n]; git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
    for p in r["patches"]: git("apply", "-3", "--index", f"{W}/v/{n}/vllm/{p}")
    out[n] = {}
    for f in r["conflicts"]:
        txt = open(f"{V}/{f}").read()
        regs = re.findall(r"^<<<<<<< .*?\n(.*?)^=======\n(.*?)^>>>>>>> .*?\n", txt, re.M | re.S)
        out[n][f] = [dict(base_lines=a.count("\n"), pr_lines=b.count("\n"), base_head=a[:300], pr_head=b[:300]) for a, b in regs]
        lines = txt.count("\n")
        print(f"### #{n} {f}: {len(regs)} region(s), file {lines} lines")
        for a, b in regs:
            print(f"  base {a.count(chr(10))} lines | PR {b.count(chr(10))} lines")
            print("  BASE> " + a[:240].replace("\n", "\n  BASE> "))
            print("  PR  > " + b[:240].replace("\n", "\n  PR  > "))
git("reset", "-q", "--hard", BASE); git("clean", "-qfd")
json.dump(out, open(f"{W}/conflict-regions.json", "w"), indent=1)
