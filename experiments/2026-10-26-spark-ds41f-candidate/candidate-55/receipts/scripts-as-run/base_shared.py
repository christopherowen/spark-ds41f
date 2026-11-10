# sp8, in ~/spark-lab/agents/round-55-57-select (run inline as python3 -): for each proposed or pending set, the
# round-1/2 base patches (candidate-54's last 30 commits, 0762..0791) that touch its non-test files.
import json,subprocess
F=json.load(open("files.json"))
V="c54-vllm"
def git(*a): return subprocess.run(["git","-C",V,*a],capture_output=True,text=True).stdout
log=git("log","--format=%h %s","round-55-57-base~30..round-55-57-base").splitlines()
assert len(log)==30
out={}
for lab in ["978","1008","988-deadrow","957-p1","957-p2","957-p4","995","956","957-p5"]:
    out[lab]={}
    for f in F[lab]:
        if f.startswith("tests/"): continue
        hits=git("log","--format=%h %s","round-55-57-base~30..round-55-57-base","--",f).splitlines()
        if hits: out[lab][f]=hits
    print(lab); [print("   ",f,"<-",[h[:60] for h in v]) for f,v in out[lab].items()]
json.dump(out,open("base-shared-files.json","w"),indent=1)
