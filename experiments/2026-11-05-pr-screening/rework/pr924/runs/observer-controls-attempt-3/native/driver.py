
from pathlib import Path
import os,sys,json,fcntl,subprocess,datetime,resource,traceback,shutil,hashlib,ast
w=Path(sys.argv[1]);pin=sys.argv[2];pr924_head='911cd4b522df2feae21b7205dc8ec911a0848a04';h=Path.home();claim=h/'spark-lab/reservations/copy-default-pr924-observer-controls.json';owned=None;locks=[]
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.nice(10)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MAX_JOBS','CMAKE_BUILD_PARALLEL_LEVEL']:os.environ[k]='1'
os.environ.update(GIT_TERMINAL_PROMPT='0',PYTHONDONTWRITEBYTECODE='1')
r={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':pin,'pr924_base_commit':pr924_head,'steps':[],'purpose':'Isolated patch application and offline CPU unit tests for PR #924 observer scope, identity and lifecycle rework','GPU_serving_or_existing_checkout_changes':False}
def save():(w/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
def run(a,name,cwd=None,required=True,stdin=None,timeout=180,env=None):
 q=subprocess.run(a,cwd=cwd,input=stdin,text=True,capture_output=True,timeout=timeout,env=env);(w/(name+'.stdout')).write_text(q.stdout);(w/(name+'.stderr')).write_text(q.stderr);r['steps'].append({'name':name,'argv':a,'exit':q.returncode,'stdout':name+'.stdout','stderr':name+'.stderr'});save()
 if required and q.returncode:raise RuntimeError(name+' exited '+str(q.returncode))
 return q
try:
 assert sys.platform=='linux' and os.uname().nodename=='spark-0016'
 c=Path('/sys/fs/cgroup')/next(x.split('::',1)[1].lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));limits={k:(c/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max']};a,b=map(int,limits['cpu.max'].split());assert a<=b and limits['memory.max']==str(4*1024**3) and limits['memory.swap.max']=='0';r['effective_limits']=limits
 assert shutil.disk_usage(w).free>4*1024**3
 for p in [h/'spark-cpu-work.lock',h/'spark-lab/offline-cpu.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for p in (h/'spark-lab/reservations').glob('*.json'):assert json.loads(p.read_text()).get('kind')=='cpu-only',str(p)
 for n in ['spark-hold.json','spark-request.json','spark3-request.json']:assert not (h/n).exists(),n
 q=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],'GPU-readonly-precondition');assert not q.stdout.strip()
 owned={'holder':'copy-default-pr924-observer-controls','kind':'cpu-only','cpus':1,'memory_gib':4,'pid':os.getpid(),'since':r['started'],'workdir':str(w),'rule':'Offline CPU observer scope/lifecycle unit tests only; no Docker/GPU/source build/install/serving/sync'}
 with claim.open('x') as f:json.dump(owned,f,indent=2)
 root=h/'spark-work/worker-test-isolation/c46-build-controls-attempt-1/checkout';base=['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=1']
 def git(*a,**kw):return run(base+['-C',str(root),*a],**kw)
 assert not git('status','--porcelain',name='clean-private-checkout').stdout.strip()
 assert git('rev-parse','HEAD',name='verify-commit').stdout.strip()==pin
 rework_dir=root/'experiments/2026-11-05-pr-screening/rework/pr924'
 meta=json.loads((rework_dir/'source-rework.json').read_text())
 diff_path=rework_dir/'observer-scope-identity-lifecycle.diff'
 assert hashlib.sha256(diff_path.read_bytes()).hexdigest()==meta['patch']['sha256']
 orig_dir=w/'original'
 iso=w/'isolated'
 iso.mkdir()
 try:
  before_hashes={}
  for rel,expected in meta['files'].items():
   if expected['original_sha256'] is not None:
    raw=(orig_dir/rel).read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    assert actual==expected['original_sha256'],f'{rel}: {actual} != {expected["original_sha256"]}'
    target=iso/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    before_hashes[rel]=actual
  (iso/'spark_ds41f/__init__.py').write_text('')
  (iso/'tests/__init__.py').write_text('')
  run(['git','apply','--check',str(diff_path)],'git-apply-check',cwd=iso)
  run(['git','apply',str(diff_path)],'git-apply',cwd=iso)
  after_hashes={}
  for rel,expected in meta['files'].items():
   target=iso/rel
   actual=hashlib.sha256(target.read_bytes()).hexdigest()
   assert actual==expected['proposed_sha256'],f'{rel}: {actual} != {expected["proposed_sha256"]}'
   ast.parse(target.read_text(),filename=rel)
   after_hashes[rel]=actual
  r['verified_hashes']={'diff_sha256':meta['patch']['sha256'],'original_sha256':before_hashes,'proposed_sha256':after_hashes}
  env=dict(os.environ,PYTHONPATH=f'{iso}:{root}')
  u1=run(['python3','-B','-m','unittest','tests.test_lossless_observer_scope','-v'],'test-lossless-observer-scope',cwd=iso,env=env,required=False)
  u2=run(['python3','-B','-m','unittest','tests.test_lossless_component_runner','-v'],'test-lossless-component-runner',cwd=iso,env=env,required=False)
  r['passed']=(u1.returncode==0 and u2.returncode==0)
 finally:
  shutil.rmtree(iso,ignore_errors=True)
  shutil.rmtree(orig_dir,ignore_errors=True)
except BaseException as e:r.update(passed=False,error=type(e).__name__+': '+str(e));(w/'failure-traceback.log').write_text(traceback.format_exc())
finally:
 if owned is not None and claim.exists() and json.loads(claim.read_text())==owned:claim.unlink()
 r['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps({k:r.get(k) for k in ['passed','error','commit','ended']}));raise SystemExit(0 if r.get('passed') else 1)
