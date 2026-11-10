# Laptop, from the scratch directory: the files each applicable set and each pending rework touches
# (files.json), the candidates' common files, and every overlapping pair. Run inline as python3 -I -.
import re,glob,json,itertools
def files(p): return set(re.findall(r'(?m)^diff --git a/(\S+)', open(p).read()))
ready={'978':'sp8/rebased/978.diff','1008':'sp8/rebased/1008.diff','988-deadrow':'sp8/rebased/988-deadrow.diff','957-p1':'sp8/rebased/957-p1.diff','957-p2':'sp8/rebased/957-p2.diff','957-p4':'sp8/rebased/957-p4.diff','967':'sp8/rebased/967.diff'}
pend={'995':glob.glob('v/995/vllm/*.patch')[0],'956':glob.glob('v/956/vllm/*.patch')[0],'957-p5':'sp8/splits/957-p5.patch','1003':glob.glob('v/1003/vllm/*.patch')[0],'953':glob.glob('v/953/vllm/*.patch')[0],'1006':glob.glob('v/1006/vllm/*.patch')[0],'957-p3':'sp8/splits/957-p3.patch','988-topk':'sp8/rebased/988-topk.diff'}
F={k:files(v) for k,v in {**ready,**pend}.items()}
cand={'55':['995','978','1008','988-deadrow'],'56':['957-p1','956'],'57':['957-p2','957-p4','957-p5']}
CF={c:set().union(*[F[x] for x in l]) for c,l in cand.items()}
for a,b in itertools.combinations(cand,2):
    print(a,b,'common:',sorted(CF[a]&CF[b]))
print()
for a,b in itertools.combinations(list(F),2):
    c=F[a]&F[b]
    if c: print(f'{a:12s} {b:12s}', sorted(c))
json.dump({k:sorted(v) for k,v in F.items()}, open('files.json','w'), indent=1)
