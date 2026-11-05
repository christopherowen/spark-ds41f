import json, os, subprocess
from pathlib import Path
root=Path.home()/'spark-lab/agents/prefix-endpoint'
os.chdir(root)
print('pid',os.getpid(),flush=True)
def run(args,cwd=root):
 print('COMMAND',args if len(args)<12 else args[:7]+['...'],flush=True)
 subprocess.run(args,cwd=cwd,check=True)
repo=root/'deployment'
run(['git','clone','--shared','--no-checkout',str(Path.home()/'projects/spark-ds41f'),str(repo)])
run(['git','remote','set-url','origin','https://github.com/christopherowen/spark-ds41f.git'],repo)
run(['git','-c','pack.threads=1','fetch','--no-tags','origin','2c4b23afcd834ba2adf6a15ca434e59fa4f49f74'],repo)
run(['git','checkout','-b','feat/prefix-state-reuse','2c4b23afcd834ba2adf6a15ca434e59fa4f49f74'],repo)
lock=json.loads((repo/'experiments/2026-10-26-spark-ds41f-candidate/candidate-38/upstreams.lock.json').read_text())
s=lock['sources']['vllm']
source=Path.home()/'projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-4158d7e5/src/vllm'
v=root/'vllm'
run(['git','clone','--shared','--no-checkout',str(source),str(v)])
run(['git','remote','set-url','origin',s['contribution_remote']],v)
run(['git','remote','add','upstream',s['upstream']],v)
run(['git','checkout','-b','prefix-endpoint',s['revision']],v)
series=repo/s['patch_series']
patches=[str((series.parent/p.strip()).resolve()) for p in series.read_text().splitlines() if p.strip() and not p.startswith('#')]
run(['git','-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','am','--committer-date-is-author-date',*patches],v)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=v,text=True).strip()
tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=v,text=True).strip()
receipt={'head':head,'tree':tree,'patches':len(patches),'matches':head==s['patch_head'] and tree==s['expected_tree']}
(root/'runs/fresh-series.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
assert receipt['matches']
run(['git','apply',str(root/'incoming/regression.patch')],v)
