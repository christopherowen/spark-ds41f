import json,subprocess,sys,re,itertools,collections
S=sys.argv[1]; V=f"{S}/vllm"
pp={int(k):v for k,v in json.load(open(f"{S}/vpatches.json")).items()}
def git(*a): return subprocess.run(["git","-C",V,*a],capture_output=True,text=True)
files={n:set(f for fn in l for f in re.findall(r'^diff --git a/(\S+)',open(fn).read(),re.M)) for n,l in pp.items()}
pairs=[(a,b) for a,b in itertools.combinations(sorted(pp),2) if files[a]&files[b]]
conf=[]
for a,b in pairs:
    git("reset","-q","--hard","HEAD"); git("clean","-qfd")
    for fn in pp[a]: git("apply","--index",fn)
    bad=None
    for fn in pp[b]:
        r=git("apply","--check",fn)
        if r.returncode: bad=sorted(set(re.findall(r'patch failed: (\S+):',r.stderr))) or r.stderr.splitlines()[:1]; break
    if bad: conf.append((a,b,bad))
git("reset","-q","--hard","HEAD"); git("clean","-qfd")
print(len(pairs),"overlapping pairs;",len(conf),"textual conflicts")
for c in conf: print(c)
json.dump(conf,open(f"{S}/conflicts.json","w"))
