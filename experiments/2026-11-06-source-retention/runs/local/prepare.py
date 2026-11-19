from pathlib import Path
import json,subprocess,hashlib,datetime,shutil
h=Path.home();dev=h/'spark-lab/checkouts/lsa-development';rel='experiments/2026-11-06-source-retention';exp=dev/rel;out=h/'spark-lab/results/source-retention/prepare-01';commands=[]
def run(argv,cwd=dev):
 commands.append({'argv':argv,'cwd':str(cwd)})
 return subprocess.run(argv,cwd=cwd,check=True,timeout=120)
def put(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write(chr(10))
base='021a464b610af717e08eef660ad5d13c9f69d8dd';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=dev,text=True).strip()==base
assert not subprocess.check_output(['git','status','--porcelain'],cwd=dev,text=True).strip()
exp.mkdir();(exp/'runs/local').mkdir(parents=True)
put(exp/'base.json',{'commit':base,'variable':'source-prefill shortlist budget512 to1024; native generation retained','runtime_parent':'ef48cc6683d177516486f4c2007ff68fa09ce235 ten-patch phase recipe','model_quality_tested':False})
put(exp/'tp4.json',{'extends':{'config':'experiments/2026-11-05-prefill-only/approximate-source-tp4.json'},'set':{'environment':{'VLLM_DS41_PAGE_SOURCE_BUDGET':'1024'}},'notes':'One variable: double source-prefill retention to65536 fine keys. Dense budget64, native generation/verification, weights and protocol remain unchanged. No exact fusion/summary interaction is included. Unpromoted; no-loss and repeated-boot request checks required.'})
graph=(dev/'experiments/2026-11-05-prefill-only/check_graph.py').read_text()
graph=graph.replace("p.add_argument('--approximate-source-prefill',action='store_true');args=p.parse_args()", "p.add_argument('--approximate-source-prefill',action='store_true');p.add_argument('--source-budget',type=int,default=512);args=p.parse_args();assert args.source_budget>0 and args.source_budget%8==0")
graph=graph.replace("'approximate_source_prefill':args.approximate_source_prefill", "'source_budget':args.source_budget,'approximate_source_prefill':args.approximate_source_prefill")
graph=graph.replace('candidate_blocks=source,row_cap=256)', 'candidate_blocks=source,budget=args.source_budget if source else None,row_cap=256)')
assert '--source-budget' in graph and 'budget=args.source_budget' in graph
(exp/'check_graph.py').write_text(graph)
registration=(dev/'experiments/2026-11-05-parallel-summary/check_registration.py').read_text().replace('candidate_blocks=source)', 'candidate_blocks=source,budget=1024 if source else 64)').replace('num_workers=4','num_workers=1').replace('parallel summary declared-call qualification','source retention declared-call qualification').replace("'cases':len(cases)","'source_budget':1024,'cases':len(cases)")
(exp/'check_registration.py').write_text(registration)
for name,source in [('graph','experiments/2026-11-05-prefill-only/sp8-source-graph/candidate.json'),('registration','experiments/2026-11-05-parallel-summary/sp8-runtime-registration/candidate.json')]:
 c=json.loads((dev/source).read_text())
 for m in c['mounts']:
  if m[1]=='/tmp/check_graph.py':m[0]=rel+'/check_graph.py'
  if m[1]=='/tmp/check_registration.py':m[0]=rel+'/check_registration.py'
  if m[1].endswith('/tilelang/compact_shortlist.py'):m[0]='experiments/2026-11-05-direct-coarse-merge/overlay/compact_shortlist.py'
 c['mounts']=[m for m in c['mounts'] if not m[1].endswith('/tilelang/fused_candidates.py')]
 c['env'].update({'OMP_NUM_THREADS':'1','MAX_JOBS':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 if name=='graph':c['argv'][-1]=c['argv'][-1].replace("'--approximate-source-prefill']", "'--approximate-source-prefill','--source-budget','1024']");assert "'1024'" in c['argv'][-1]
 (exp/('sp8-'+name)).mkdir();put(exp/('sp8-'+name)/'candidate.json',c)
receipt=json.loads((dev/'experiments/2026-11-05-prefill-only/runs/local/source-check.json').read_text());lock=json.loads((dev/'experiments/2026-10-26-spark-ds41f-candidate/candidate-36/upstreams.lock.json').read_text())['sources']['vllm'];cached=h/'spark-lab/checkouts/vllm-drafter-orchestration-20261105';fresh=dev/'.work/source-retention-verification-01'
assert subprocess.check_output(['git','rev-parse',receipt['base']+'^{tree}'],cwd=cached,text=True).strip()==lock['expected_tree']
run(['git','clone','--shared','--no-checkout',str(cached),str(fresh)])
run(['git','remote','rename','origin','upstream'],fresh);run(['git','remote','set-url','upstream',lock['upstream']],fresh);run(['git','remote','add','origin',lock['contribution_remote']],fresh);run(['git','checkout','-b','source-retention-verification',receipt['base']],fresh)
for p,digest in receipt['ordered_patches'].items():
 assert hashlib.sha256((dev/p).read_bytes()).hexdigest()==digest
 run(['git','apply','--check',str(dev/p)],fresh);run(['git','apply',str(dev/p)],fresh)
for p,target in receipt['exact_runtime_files'].items():assert (dev/p).read_bytes()==(fresh/target).read_bytes()
run(['git','diff','--check'],fresh);run(['python3','experiments/2026-11-05-prefill-only/test_dispatch.py']);run(['bin/spark','--cluster-config',rel+'/tp4.json','doctor']);run(['git','diff','--check'])
put(exp/'runs/local/source-check.json',{**receipt,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':'source-retention-verification','source_budget':1024,'new_runtime_code':False,'new_kernels':False,'commands':commands,'cpu_quota_percent':100,'memory_limit_gib':4,'build_workers':1,'complete_image_build_inputs_present':False})
shutil.copyfile(out/'prepare.py',exp/'runs/local/prepare.py')
(exp/'README.md').write_text('# Higher source-prefill retention'+chr(10)+chr(10)+'Base `'+base+'`. One intended variable relative to the rejected compact-source-prefill/native-generation recipe: source budget512 to1024, retaining65536 fine keys. Dense budget64, native generation/verification, cache geometry, weights, graders and response policy stay unchanged. No fusion or cooperative maintenance interaction is included. This is a generic retention change, with no task, input ID or answer-specific serving logic.'+chr(10)+chr(10)+'The512 parent loses a previously correct128K QA answer; native generation did not repair it. Higher retention is a hypothesis to test, not evidence of quality recovery. The fresh ten-patch source application equals the unchanged runtime overlay. Commands and preparation code remain in `runs/local`. GPU declared-call registration and72 strict graph routes, including empty/restored extents and source handoff, must pass from one clean published commit before model use. Captured prefill queries do not establish actual decode performance or final-answer quality.'+chr(10)+chr(10)+'Model tests must reject every lost independently correct answer and every newly incomplete response, and require two fresh candidate boots before comparing speed. Temperature0, seed0, high thinking and65536 generated tokens stay shared. All thinking and unsuccessful request time counts. Raw model text, token IDs, programs and activations remain private on nodes. Remote preparation uses one CPU,4 GiB and one worker; GPU work uses native holds and explicit limits. Full image-build inputs are unavailable. No quality win, speed win or production promotion is claimed.'+chr(10))
rejection=h/'spark-lab/results/prefill-fusion/prepare-01/parent-quality-rejection.json';shutil.copyfile(rejection,dev/'experiments/2026-11-06-prefill-fusion/runs/local/parent-quality-rejection.json')
p=dev/'experiments/2026-11-06-prefill-fusion/README.md';p.write_text(p.read_text()+chr(10)+'The parent phase policy subsequently loses the128K QA answer in `source-prefill-ruler-window-01`. Its queued integration GPU check is cancelled before starting. The exact optimization combination remains prepared but model-unqualified; it cannot claim to recover this selection loss.'+chr(10))
run(['git','add',rel,'experiments/2026-11-06-prefill-fusion']);run(['git','diff','--cached','--check']);run(['git','-c','user.name=Christopher Owen','-c','user.email=3221756+christopherowen@users.noreply.github.com','commit','-m','Test higher source-prefill retention after answer rejection'])
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dev,text=True).strip();bundle=out/'prepared.bundle';run(['git','bundle','create',str(bundle),base+'..HEAD']);put(out/'prepared.json',{'commit':head,'bundle_bytes':bundle.stat().st_size,'source_verified':True,'published':False});print((out/'prepared.json').read_text())
