
from pathlib import Path
import os,sys,json,fcntl,subprocess,datetime,resource,traceback,shutil
w=Path(sys.argv[1]);pin=sys.argv[2];h=Path.home();claim=h/'spark-lab/reservations/copy-default-c46-build-controls.json';owned=None;locks=[]
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));os.nice(10)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MAX_JOBS','CMAKE_BUILD_PARALLEL_LEVEL']:os.environ[k]='1'
os.environ.update(GIT_TERMINAL_PROMPT='0',PYTHONDONTWRITEBYTECODE='1')
r={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':pin,'steps':[],'purpose':'offline build-resource controls and static profile doctor','GPU_serving_or_existing_checkout_changes':False}
def save():(w/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
def run(a,name,cwd=None,required=True,stdin=None,timeout=180):
 q=subprocess.run(a,cwd=cwd,input=stdin,text=True,capture_output=True,timeout=timeout);(w/(name+'.stdout')).write_text(q.stdout);(w/(name+'.stderr')).write_text(q.stderr);r['steps'].append({'name':name,'argv':a,'exit':q.returncode,'stdout':name+'.stdout','stderr':name+'.stderr'});save()
 if required and q.returncode:raise RuntimeError(name+' exited '+str(q.returncode))
 return q
try:
 assert sys.platform=='linux' and os.uname().nodename=='spark-0016'
 c=Path('/sys/fs/cgroup')/next(x.split('::',1)[1].lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));limits={k:(c/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max']};a,b=map(int,limits['cpu.max'].split());assert a<=b and limits['memory.max']==str(8*1024**3) and limits['memory.swap.max']=='0';r['effective_limits']=limits
 assert shutil.disk_usage(w).free>4*1024**3
 for p in [h/'spark-cpu-work.lock',h/'spark-lab/offline-cpu.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for p in (h/'spark-lab/reservations').glob('*.json'):assert json.loads(p.read_text()).get('kind')=='cpu-only',str(p)
 for n in ['spark-hold.json','spark-request.json','spark3-request.json']:assert not (h/n).exists(),n
 q=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],'GPU-readonly-precondition');assert not q.stdout.strip()
 owned={'holder':'copy-default-c46-build-controls','kind':'cpu-only','cpus':1,'memory_gib':8,'pid':os.getpid(),'since':r['started'],'workdir':str(w),'rule':'Offline unit/source checks only; no Docker run, GPU tests, source builds, dependency installation or serving operations'}
 with claim.open('x') as f:json.dump(owned,f,indent=2)
 root=w/'checkout';base=['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=1']
 run(base+['clone','--no-checkout','--filter=blob:none','--depth=1','--single-branch','--branch','feat/remote-source-proof','--origin','deployment','--reference-if-able',str(h/'projects/spark-ds41f'),'https://github.com/christopherowen/spark-ds41f.git',str(root)],'clone-sparse-packet')
 def git(*a,**kw):return run(base+['-C',str(root),*a],**kw)
 git('sparse-checkout','init','--no-cone',name='sparse-init')
 patterns=['/.gitignore','/AGENTS.md','/README.md','/pyproject.toml','/upstreams.lock.json','/bin/','/spark_ds41f/','/scripts/','/tests/','/config/','/docker/','/requirements/','/patches/','/manifests/','/docs/','/experiments/*/*.json','/experiments/**/source.json','/experiments/**/upstreams.lock.json','/experiments/**/series','/experiments/**/*.patch','/experiments/2026-10-26-spark-ds41f-candidate/candidate-*/*.json','/experiments/2026-11-06-worker-test-isolation-rework/*.py']
 git('sparse-checkout','set','--no-cone','--stdin',name='sparse-patterns',stdin='\n'.join(patterns)+'\n')
 git('checkout','-b','feat/c46-build-control-check',pin,name='exact-published-checkout')
 got=git('rev-parse','HEAD',name='verify-commit').stdout.strip();assert got==pin
 r['sparse_patterns']=patterns;r['existing_reference_checkout_untouched']=True
 q=run(['python3','-B','-m','unittest','discover','-s','tests','-p','test_build.py','-v'],'build-unit-tests',cwd=root,required=False)
 r['unit_exit']=q.returncode
 if q.returncode:raise RuntimeError('build unit checks failed; preserve original diagnostics')
 # Only static checks; missing ignored site maps are retained as fixture setup
 # evidence and never copied to any serving checkout by this job.
 for label,profile in [('tp3','tp3.json'),('tp4','tp4.json')]:
  q=run(['python3','-B','bin/spark','--cluster-config','experiments/2026-10-26-spark-ds41f-candidate/candidate-46/'+profile,'doctor'],'doctor-'+label,cwd=root,required=False)
  r['doctor_'+label+'_exit']=q.returncode
 git('diff','--check',name='whitespace')
 r['passed']=r['unit_exit']==0 and r.get('doctor_tp3_exit')==0 and r.get('doctor_tp4_exit')==0
except BaseException as e:r.update(passed=False,error=type(e).__name__+': '+str(e));(w/'failure-traceback.log').write_text(traceback.format_exc())
finally:
 if owned is not None and claim.exists() and json.loads(claim.read_text())==owned:claim.unlink()
 r['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps({k:r.get(k) for k in ['passed','error','unit_exit','doctor_tp3_exit','doctor_tp4_exit','commit','ended']}));raise SystemExit(0 if r.get('passed') else 1)
