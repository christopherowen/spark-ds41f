import json, pathlib, subprocess
root=pathlib.Path('/home/swank/spark-lab/collective-pdl/deployment'); out=root/'experiments/2026-11-05-collective-pdl'; (out/'runs').mkdir(parents=True,exist_ok=True)
pin='experiments/2026-10-26-spark-ds41f-candidate/candidate-38/upstreams.lock.json'
lock=json.loads((root/pin).read_text()); s=lock['sources']['vllm']; dest=root/'.work/vllm'; dest.parent.mkdir(exist_ok=True)
caches=sorted(pathlib.Path('/home/swank/projects/spark-ds41f/.work/build').glob('vllm-19f*/src/vllm'))
assert caches, 'no existing vLLM source cache'
with (out/'runs/fresh-vllm-series.log').open('w') as log:
 def run(args,cwd=None): return subprocess.run(args,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
 run(['git','clone','--shared','--no-checkout',str(caches[0]),str(dest)])
 run(['git','remote','remove','origin'],dest);run(['git','remote','add','upstream',s['upstream']],dest);run(['git','remote','add','origin',s['contribution_remote']],dest)
 run(['git','switch','-c','candidate-38-collective-audit',s['revision']],dest)
 series=root/s['patch_series']; patches=[str((series.parent/p.strip()).resolve()) for p in series.read_text().splitlines() if p.strip() and not p.lstrip().startswith('#')]
 run(['git','-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','am','--quiet','--committer-date-is-author-date',*patches],dest)
 tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=dest,text=True).strip(); assert tree==s['expected_tree'],(tree,s['expected_tree'])
 result={'command':'python3 reproduce.py','host':'dgx4','patch_count':len(patches),'tree':tree,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip(),'expected_tree':s['expected_tree'],'passed':True}
 (out/'runs/fresh-vllm-series.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
