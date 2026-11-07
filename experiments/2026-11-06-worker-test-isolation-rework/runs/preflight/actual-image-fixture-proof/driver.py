from pathlib import Path
import os,json,subprocess,datetime,sys,hashlib,base64,fcntl,resource,signal,shlex,traceback
w=Path(sys.argv[1]);p=json.loads((w/'payload.json').read_text());h=Path.home();claim=h/'spark-lab/reservations/copy-default-c46-cpu-fixture.json';claim_data=None;locks=[]
image='sha256:ac3c5a2ca2e934295ca7be60d69566d4c406aff22057b0a911661d49e5664905';r={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'actual-image CPU pytest fixture semantics only','steps':[],'GPU_native_gate_serving_or_checkout_changes':False,'published_commit':p['published_commit'],'source_head':p['head'],'source_tree':p['tree']}
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
def save():(w/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
def run(a,name,timeout=20):
 q=subprocess.run(a,capture_output=True,text=True,timeout=timeout);(w/(name+'.stdout')).write_text(q.stdout);(w/(name+'.stderr')).write_text(q.stderr);r['steps'].append({'name':name,'argv':a,'exit':q.returncode,'stdout_sha256':hashlib.sha256(q.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(q.stderr.encode()).hexdigest()});save()
 if q.returncode:raise RuntimeError(name+' exited '+str(q.returncode))
 return q.stdout
def shared():
 code="from pathlib import Path;import json;h=Path.home();print(json.dumps({n:json.loads((h/n).read_text()) if (h/n).exists() else None for n in ['spark-hold.json','spark-request.json']}))"
 q=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','dgx1','python3','-c',shlex.quote(code)],capture_output=True,text=True,timeout=10)
 if q.returncode:raise RuntimeError('head coordination read failed: '+q.stderr)
 d=json.loads(q.stdout);hold=d['spark-hold.json']
 if hold:
  scope=hold.get('resource_scope',{});nodes=scope.get('nodes')
  if not nodes or scope.get('holder')!=hold.get('holder') or scope.get('token')!=hold.get('window') or 'dgx4' in nodes:raise RuntimeError('head hold is unscoped or owns DGX4; defer')
 return d
def host_idle():
 for n in ['spark-hold.json','spark-request.json','spark3-request.json']:
  if (h/n).exists():raise RuntimeError('local hold/request present; defer')
 for f in (h/'spark-lab/reservations').glob('*.json'):
  if f!=claim:raise RuntimeError('foreign local host claim; defer')
 if subprocess.check_output(['docker','ps','-q'],text=True).strip():raise RuntimeError('DGX4 host has active containers; defer')
 q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=8)
 if q.returncode or q.stdout.strip():raise RuntimeError('GPU/management readiness differs; no CPU container started')
 mem=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 if mem<8*1024**2:raise RuntimeError('host CPU-proof memory floor failed')
try:
 cg=Path('/sys/fs/cgroup')/next(x.split('::',1)[1].lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));r['host_cgroup']={n:(cg/n).read_text().strip() for n in ['cpu.max','memory.max','memory.swap.max']};a,b=map(int,r['host_cgroup']['cpu.max'].split());assert a<=b and r['host_cgroup']['memory.max']==str(1024**3) and r['host_cgroup']['memory.swap.max']=='0'
 for f in [h/'spark-cpu-work.lock',h/'spark-lab/offline-cpu.lock']:
  f.parent.mkdir(parents=True,exist_ok=True);o=f.open('a');fcntl.flock(o,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(o)
 r['shared_before']=shared();host_idle()
 claim_data={'holder':'copy-default-c46-cpu-fixture','kind':'cpu-only','nodes':['dgx4'],'token':'c46-cpu-fixture-attempt-1','cpus':2,'memory_gib':3,'since':r['started'],'pid':os.getpid(),'controller_commit':p['published_commit'],'controller':'fixture_witness.py plus recorded CPU-only operator guard','watchdog':'user systemd RuntimeMaxSec120; each container SIGALRM30s','restore':'no serving mutation; CPU containers automatically removed','rule':'CPU fixture proof only, no GPU/serving/sync or shared hold/request mutation'}
 with claim.open('x') as f:json.dump(claim_data,f,indent=2)
 for name,item in p['source'].items():
  assert name in p['hashes'] and not Path(name).is_absolute() and '..' not in Path(name).parts
  for phase in ['before','after']:
   raw=base64.b64decode(item[phase] or item['before']);assert hashlib.sha256(raw).hexdigest()==p['hashes'][name][phase+'_sha256'];dest=w/(phase+'-source')/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 raw=base64.b64decode(p['witness']);assert hashlib.sha256(raw).hexdigest()==p['witness_sha256'];(w/'witness.py').write_bytes(raw)
 inspections=json.loads(run(['docker','image','inspect',image],'image-before'));assert len(inspections)==1 and inspections[0]['Id']==image
 labels=inspections[0]['Config'].get('Labels',{});assert labels['local.spark-ds41f.vllm.patch-head']=='99029e6cbdb966ff198dd13902a0825e8d507e27' and labels['local.spark-ds41f.vllm.tree']=='082bda0f13e94290f462eaf55d143cdb7931f9e5';r['frozen_image_identity']={'image':image,'labels':labels};save()

 (w/'guard.py').write_text(p['guard_code']);(w/'hash-manifest.json').write_text(json.dumps(p['hashes']))
 r['container_guards']=[]
 for phase,args in [('metadata',['--metadata-only']),('witness',['--before-source','/proof/before-source','--after-source','/proof/after-source'])]:
  shared();host_idle();name='c46-fixture-'+phase+'-attempt-1'
  a=['docker','run','--rm','--pull','never','--name',name,'--cidfile',str(w/(phase+'.cid')),'--runtime','runc','--read-only','--network','none','--cpus','1','--memory','2g','--memory-swap','2g','--pids-limit','64','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=64m']
  for k,v in {'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MAX_JOBS':'1','EXPECTED_IMAGE_ID':image}.items():a+=['-e',k+'='+v]
  a+=['--mount','type=bind,src='+str(w)+',dst=/proof,readonly','--entrypoint','python3',image,'-B','/proof/guard.py',*args]
  output=run(a,phase,timeout=45);items=[json.loads(line) for line in output.splitlines() if line.startswith('{')];assert any(x.get('kind')=='actual_CPU_container_resource_and_runtime_guard' for x in items);assert any(x.get('status')=='captured' and x.get('kind')=='actual_image_pytest_metadata' for x in items)
  if phase=='witness':assert any(x.get('status')=='source_fixture_cleanup_witness_passed' for x in items)
  r['container_guards'].append({'phase':phase,'container_id':(w/(phase+'.cid')).read_text().strip(),'records':items});save()
 after=json.loads(run(['docker','image','inspect',image],'image-after'));assert after[0]['Id']==image and after[0]['Config']['Labels']==labels;r['shared_after']=shared();r['passed']=True
except BaseException as e:r.update(passed=False,error=type(e).__name__+': '+str(e));(w/'failure-traceback.log').write_text(traceback.format_exc())
finally:
 if claim_data is not None and claim.exists() and json.loads(claim.read_text())==claim_data:claim.unlink()
 r['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps({'passed':r.get('passed'),'error':r.get('error'),'ended':r['ended'],'containers':len(r.get('container_guards',[]))}));raise SystemExit(0 if r.get('passed') else 1)
