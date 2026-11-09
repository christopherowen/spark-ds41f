import json,subprocess,os,sys,re
S=sys.argv[1]; R="/Users/christopherowen/projects/spark-ds41f"; V=f"{S}/vllm"
d={p['number']:p for p in json.load(open(f"{S}/prs.json"))}
def git(*a,cwd=R,inp=None): return subprocess.run(["git","-C",cwd,*a],capture_output=True,text=True,input=inp)
pp={}
for n,p in d.items():
    fs=[f['path'] for f in p['files'] if f['path'].endswith('.patch') and '/vllm/' in '/'+f['path'] and re.match(r'07\d\d-|0045-',f['path'].split('/')[-1])]
    fs=[f for f in fs if f.startswith('experiments/')] or fs
    if not fs: continue
    os.makedirs(f"{S}/v/{n}",exist_ok=True); lst=[]
    for f in sorted(fs, key=lambda x:x.split('/')[-1]):
        fn=f"{S}/v/{n}/"+f.split('/')[-1]; open(fn,"w").write(git("show",f"refs/review/pr-{n}:{f}").stdout); lst.append(fn)
    pp[n]=lst
json.dump(pp,open(f"{S}/vpatches.json","w"),indent=1)
bad={}
for n,lst in sorted(pp.items()):
    git("reset","-q","--hard","HEAD",cwd=V); git("clean","-qfd",cwd=V)
    ok=True
    for fn in lst:
        r=git("apply","--index",fn,cwd=V)
        if r.returncode: ok=False; bad[n]=r.stderr.strip().splitlines()[:2]; break
    print(n,"OK" if ok else "FAIL "+str(bad[n]), len(lst))
git("reset","-q","--hard","HEAD",cwd=V); git("clean","-qfd",cwd=V)
