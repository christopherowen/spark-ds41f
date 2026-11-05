import json,pathlib,subprocess
repo=pathlib.Path('/home/swank/spark-lab/collective-pdl/deployment')
exp=repo/'experiments/2026-11-05-collective-pdl'
manifest=json.loads((exp/'source.json').read_text())
cache=pathlib.Path('/home/swank/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-05dd5db6')
dest=repo/'.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-f970f8b7'
records=[]
for name in ['vllm','b12x','nccl','tilelang','tile_kernels','cutlass']:
    source=cache/'src'/name; target=dest/'src'/name
    assert not target.exists(), target
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=source,text=True).strip(),name
    head,tree=subprocess.check_output(['git','rev-parse','HEAD','HEAD^{tree}'],cwd=source,text=True).splitlines()
    if name in manifest:
        assert head==manifest[name]['patch_head'] and tree==manifest[name]['expected_tree'],name
    else:
        lock=json.loads((exp/'upstreams.lock.json').read_text())
        assert head==lock['sources'][name]['revision'],name
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cp','--reflink=auto','--archive',str(source),str(target)],check=True)
    records.append({'project':name,'cache':str(source),'head':head,'tree':tree})
source=cache/'context/cutlass-dsl-wheels'; target=dest/'context/cutlass-dsl-wheels'
assert not target.exists(),target
target.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['cp','--reflink=auto','--archive',str(source),str(target)],check=True)
(exp/'runs/cached-inputs.json').write_text(json.dumps({'method':'Copy clean cached checkouts, verify exact pinned heads and trees, then run canonical build prepare and check; cached states are not authority.','sources':records},indent=2)+'\n')
print('Seeded cached inputs',flush=True)
