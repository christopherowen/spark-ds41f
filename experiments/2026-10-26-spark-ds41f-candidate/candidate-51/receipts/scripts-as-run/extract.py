# Extract every pool PR's patch files from its fetched head (refs/review/pr-<N>) into v/<N>/<component>/,
# and record targets, commit count per patch, merge-base distance and head SHA.
import json, subprocess, re, collections, sys, os
S = sys.argv[1]; R = f"{S}/repo"
d = {p['number']: p for p in json.load(open(f"{S}/prs.json"))}
pool = [int(x) for x in open(f"{S}/pool.txt").read().split()]
def git(*a): return subprocess.run(["git", "-C", R, *a], capture_output=True, text=True).stdout
out = {}
for n in pool:
    p = d[n]; ref = f"refs/review/pr-{n}"
    head = git("rev-parse", ref).strip()
    mb = git("merge-base", ref, "origin/main").strip()
    behind = len(git("rev-list", f"{mb}..origin/main").split())
    pats = [f['path'] for f in p['files'] if f['path'].endswith('.patch')]
    ent = []
    for pf in sorted(pats):
        body = git("show", f"{ref}:{pf}")
        comp = pf.split('/')[-2]
        if comp.startswith('20') or comp == 'capture':  # patch directly under the experiment folder
            comp = 'sparknet' if 'sparknet' in pf.split('/')[-1] or 'oneshot' in pf.split('/')[-1] else comp
        dst = f"{S}/v/{n}/{comp}"; os.makedirs(dst, exist_ok=True)
        fn = f"{dst}/{pf.split('/')[-1]}"; open(fn, "w").write(body)
        tg = sorted(set(re.findall(r'^diff --git a/(\S+)', body, re.M)))
        ncommits = len(re.findall(r'^From [0-9a-f]{40} ', body, re.M))
        subj = re.findall(r'^Subject: (.*)$', body, re.M)
        ent.append(dict(path=pf, component=comp, local=fn, targets=tg, commits=ncommits, subjects=subj,
                        lines=body.count('\n'), adds=len(re.findall(r'^\+(?!\+\+)', body, re.M))))
    out[n] = dict(title=p['title'], author=p['author']['login'], head=head, merge_base=mb, behind_main=behind, patches=ent)
json.dump(out, open(f"{S}/scan.json", "w"), indent=1)
for n, o in out.items():
    print(n, o['head'][:12], 'behind', o['behind_main'], [(e['component'], os.path.basename(e['path'])[:50], e['commits'], len(e['targets']), e['adds']) for e in o['patches']])
