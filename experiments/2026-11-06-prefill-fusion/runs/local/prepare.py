from pathlib import Path
import json,subprocess,hashlib,datetime
h=Path.home();dev=h/'spark-lab/checkouts/lsa-development';rel='experiments/2026-11-06-prefill-fusion';exp=dev/rel;out=h/'spark-lab/results/prefill-fusion/prepare-01'
def run(args,cwd=dev):return subprocess.run(args,cwd=cwd,check=True,timeout=120)
base='61eaf51bb93d72cf2da3d79bddaab05cfe717a85';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=dev,text=True).strip()==base
assert not subprocess.check_output(['git','status','--porcelain'],cwd=dev,text=True).strip()
exp.mkdir();(exp/'runs/local').mkdir(parents=True)
def put(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write(chr(10))
put(exp/'base.json',{'commit':base,'variable':'interaction of exact candidate fusion and cooperative summary maintenance with the same compact-source-prefill/native-generation phase policy','runtime_adopted':False,'quality_changes_intended':False})
parent='experiments/2026-11-05-prefill-only/approximate-source-tp4.json'
mounts=json.loads((dev/'experiments/2026-11-05-parallel-summary/tp4.json').read_text())['set']['container']['mounts']
for m in mounts:
 if m[1].endswith('/tilelang/attention.py'):m[0]='{checkout}/experiments/2026-11-05-prefill-only/overlay/attention.py'
put(exp/'tp4.json',{'extends':{'config':parent},'set':{'container':{'mounts':mounts}},'notes':'Explicit interaction: existing exact final-candidate fusion plus byte-identical cooperative summary construction over the unchanged source-prefill/native-generation recipe. Same budgets, cache geometry, native fine arithmetic and protocol. No quality or speed qualification claimed.'})
for name,source in [('graph','experiments/2026-11-05-prefill-only/sp8-source-graph/candidate.json'),('registration','experiments/2026-11-05-parallel-summary/sp8-runtime-registration/candidate.json'),('actual','experiments/2026-11-05-parallel-summary/sp8-runtime-actual/candidate.json')]:
 candidate=json.loads((dev/source).read_text())
 for m in candidate['mounts']:
  if m[1].endswith('/tilelang/compact_shortlist.py'):m[0]='experiments/2026-11-05-parallel-summary/overlay/compact_shortlist.py'
  if m[1].endswith('/tilelang/attention.py'):m[0]='experiments/2026-11-05-prefill-only/overlay/attention.py'
 if not any(m[1].endswith('/tilelang/fused_candidates.py') for m in candidate['mounts']):candidate['mounts'].append(['experiments/2026-11-05-fused-candidates/overlay/fused_candidates.py','/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/fused_candidates.py'])
 candidate['env'].update({'OMP_NUM_THREADS':'1','MAX_JOBS':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 (exp/('sp8-'+name)).mkdir();put(exp/('sp8-'+name)/'candidate.json',candidate)
phase=json.loads((dev/'experiments/2026-11-05-prefill-only/runs/local/source-check.json').read_text());parallel=json.loads((dev/'experiments/2026-11-05-parallel-summary/runs/local/source-check.json').read_text())
patches=dict(phase['ordered_patches']);extra={k:v for k,v in parallel['ordered_patches'].items() if k not in patches};assert len(extra)==2;patches.update(extra);assert len(patches)==12
lock=json.loads((dev/'experiments/2026-10-26-spark-ds41f-candidate/candidate-36/upstreams.lock.json').read_text())['sources']['vllm'];cached=h/'spark-lab/checkouts/vllm-drafter-orchestration-20261105';fresh=dev/'.work/prefill-fusion-verification-01'
assert subprocess.check_output(['git','rev-parse',phase['base']+'^{tree}'],cwd=cached,text=True).strip()==lock['expected_tree']
run(['git','clone','--shared','--no-checkout',str(cached),str(fresh)])
run(['git','remote','rename','origin','upstream'],fresh);run(['git','remote','set-url','upstream',lock['upstream']],fresh);run(['git','remote','add','origin',lock['contribution_remote']],fresh)
run(['git','checkout','-b','prefill-fusion-verification',phase['base']],fresh)
for p,digest in patches.items():
 assert hashlib.sha256((dev/p).read_bytes()).hexdigest()==digest
 run(['git','apply','--check',str(dev/p)],fresh);run(['git','apply',str(dev/p)],fresh)
files={**phase['exact_runtime_files'],**parallel['exact_runtime_files'],'experiments/2026-11-05-fused-candidates/overlay/fused_candidates.py':'vllm/models/deepseek_v4_1/tilelang/fused_candidates.py'}
for source,target in files.items():assert (dev/source).read_bytes()==(fresh/target).read_bytes()
run(['git','diff','--check'],fresh)
put(exp/'runs/local/source-check.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':phase['base'],'tree':lock['expected_tree'],'branch':'prefill-fusion-verification','ordered_patches':patches,'exact_runtime_files':files,'new_kernels':False,'new_approximation':False,'diff_check':True,'cpu_limit':1,'memory_limit_gib':4,'complete_image_build_inputs_present':False})
(exp/'README.md').write_text('# Exact prefill pipeline cleanup'+chr(10)+chr(10)+'Base `'+base+'`. This explicitly tests the interaction of two previously component-qualified exact optimizations: final-candidate fusion and cooperative summary construction. The phase policy stays compact dense/source prefill with native generation and verification. Budgets, cache geometry, weights, fine-score arithmetic and answer protocol remain unchanged. No input, task ID or answer influences serving policy.'+chr(10)+chr(10)+'A fresh named source tree applies the ten phase-routing patches followed by the existing fusion and summary patches. Runtime attention, compact selection and candidate-fusion files must equal the overlays byte for byte. No kernel body is changed. GPU graph replay, declared-call registration and actual-input checks remain required on a clean published commit, followed by independent final-answer gates and two fresh model boots. This does not inherit an untested quality or request-speed claim from a component timing.'+chr(10)+chr(10)+'Exact commands, source hashes and preparation output remain in `runs/local`. GPU work is queued through native holds and uses separate clean runner checkouts. One CPU, 4 GiB and one build worker cap remote preparation. Raw text, token IDs, activations and programs remain node-local. All model thinking and unsuccessful time counts; temperature0, seed0, high thinking and65536 generated tokens stay shared. No quality or performance result and no production promotion are claimed.'+chr(10))
run(['python3','experiments/2026-11-05-prefill-only/test_dispatch.py'])
run(['bin/spark','--cluster-config',rel+'/tp4.json','doctor'])
run(['git','diff','--check'])
run(['git','add',rel]);run(['git','diff','--cached','--check'])
run(['git','-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','commit','-m','Prepare exact prefill fusion and summary integration'])
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dev,text=True).strip();bundle=out/'prepared.bundle';run(['git','bundle','create',str(bundle),base+'..HEAD'])
put(out/'prepared.json',{'commit':head,'bundle_bytes':bundle.stat().st_size,'published':False,'source_verified':True,'ordered_patch_count':12})
print((out/'prepared.json').read_text())
