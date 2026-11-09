import json,subprocess,re,collections,sys
S=sys.argv[1]
d=json.load(open(f"{S}/prs.json"))
def git(*a): return subprocess.run(["git",*a],capture_output=True,text=True).stdout
out={}
for p in sorted(d,key=lambda p:p['number']):
    n=p['number']; ref=f"refs/review/pr-{n}"
    mb=git("merge-base",ref,"origin/main").strip()
    behind=len(git("rev-list",f"{mb}..origin/main").split())
    patches=[f['path'] for f in p['files'] if f['path'].endswith('.patch')]
    targets=collections.defaultdict(set); nums=[]
    for pf in patches:
        m=re.match(r'(\d{4})-',pf.split('/')[-1]);
        if m: nums.append(m.group(1))
        comp=pf.split('/')[-2]
        body=git("show",f"{ref}:{pf}")
        for t in re.findall(r'^diff --git a/(\S+)',body,re.M):
            targets[comp].add(t)
    out[n]=dict(title=p['title'],author=p['author']['login'],behind=behind,patch_nums=sorted(set(nums)),npatches=len(patches),targets={k:sorted(v) for k,v in targets.items()},head=p['headRefName'])
json.dump(out,open(f"{S}/scan.json","w"),indent=1)
for n,o in out.items():
    print(n,o['behind'],o['patch_nums'],{k:len(v) for k,v in o['targets'].items()})
