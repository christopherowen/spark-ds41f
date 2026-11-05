"""Read-only receipt for the original candidate-36 service after the window closes."""
import base64,concurrent.futures,datetime,hashlib,importlib.util,inspect,json,pathlib,re,shlex,subprocess,sys
repo=pathlib.Path.cwd();private=pathlib.Path(__file__).resolve().parent.parent
SSH_ROUTE=['-o', 'HostName=10.0.1.71', '-o', 'ProxyJump=dgx2', '-o', 'HostKeyAlias=dgx1', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=1']
PIN='cb657ab1f5dba4e7cca48a48b0c2fe4e8d4a711f'
IMAGE='sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11'
PROFILE='experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json'
source=repo/'experiments/2026-11-03-candidate-38-reproduction/collect_runtime.py'
assert source.read_bytes()==subprocess.check_output(['git','show',PIN+':'+str(source.relative_to(repo))])
module=importlib.util.spec_from_file_location('recorded_collector',source);m=importlib.util.module_from_spec(module);module.loader.exec_module(m)
function=inspect.getsource(m.collect_on_node)
payload=function+'''
import json,pathlib,subprocess
r=collect_on_node('spark-ds41f',100000,33554432)
p=pathlib.Path.home()/'projects/spark-ds41f'
r['checkout']={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True),'branch':subprocess.run(['git','symbolic-ref','--quiet','--short','HEAD'],cwd=p,text=True,capture_output=True).stdout.strip()}
r['fan_service']=subprocess.run(['systemctl','is-active','dgx-fan-control.service'],text=True,capture_output=True).stdout.strip()
r['hold_files']={name:(pathlib.Path.home()/name).exists() for name in ['spark-hold.json','spark3-hold.json']}
r['request_files']={name:(pathlib.Path.home()/name).exists() for name in ['spark-request.json','spark3-request.json']}
r['lab_clock_lock_marker']=(pathlib.Path.home()/'spark-lab/gpu-clocks-locked.json').exists()
print(json.dumps(r))
'''
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=private/'restoration'/stamp;out.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fault=re.compile(r'ERROR|CRITICAL|Traceback|device.side assert|CUDA error|illegal memory access|gate logits contain|EngineCore.*died',re.I)
def one(node):
 directory=out/node;directory.mkdir();result=subprocess.run(['ssh',*(SSH_ROUTE if node=='dgx1' else []),node,'python3 -c '+shlex.quote(payload)],text=True,capture_output=True)
 if result.returncode:
  record={'node':node,'issues':['remote collection exited '+str(result.returncode)]}
 else:
  record=json.loads(result.stdout);record['node']=node;log=base64.b64decode(record.pop('log_base64',''),validate=True);(directory/'engine.log').write_bytes(log)
  issues=[record['error']] if record.get('error') else []
  c=record.get('container',{});image=record.get('image',{})
  if c.get('image_id')!=IMAGE or image.get('id')!=IMAGE:issues.append('original image differs')
  if c.get('profile')!=PROFILE:issues.append('original profile differs')
  if c.get('status')!='running' or record.get('status_after_collection')!='running' or not record.get('same_container_after_collection') or c.get('restart_count')!=0:issues.append('container unhealthy or changed')
  if not log or record.get('log',{}).get('truncated') is not False or hashlib.sha256(log).hexdigest()!=record.get('log',{}).get('sha256'):issues.append('log missing, truncated or hash mismatch')
  if fault.search(log.decode(errors='replace')):issues.append('fault pattern in restored rank log')
  if record.get('checkout')!={'head':PIN,'status':'','branch':''}:issues.append('checkout differs or is dirty')
  if record.get('fan_service')!=('inactive' if node=='dgx3' else 'active'):issues.append('fan service differs from initial state')
  if node=='dgx1' and any(record.get('hold_files',{}).values()):issues.append('cluster window is held')
  if node=='dgx1' and record.get('lab_clock_lock_marker'):issues.append('lab clock lock remains recorded')
  record['issues']=issues
 (directory/'runtime.json').write_text(json.dumps(record,indent=2)+'\n');return record
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(one,['dgx1','dgx2','dgx3']))
fingerprints={(tuple(r.get('image',{}).get('layers',[])),r.get('image',{}).get('config_sha256')) for r in records}
issues={r['node']:r['issues'] for r in records if r['issues']}
if len(fingerprints)!=1:issues['image_agreement']=['Config or layers differ']
(out/'collector-function.py').write_text(function)
summary={'profile':PROFILE,'expected_image_id':IMAGE,'expected_checkout':PIN,'issues':issues,'passed':not issues,'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'collector_source_sha256':sha(source),'collector_function_sha256':sha(out/'collector-function.py'),'wrapper_sha256':sha(pathlib.Path(__file__)),'records':{r['node']:{'container':r.get('container'),'image':r.get('image'),'fan_service':r.get('fan_service'),'hold_files':r.get('hold_files'),'request_files':r.get('request_files'),'lab_clock_lock_marker':r.get('lab_clock_lock_marker')} for r in records},'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(str(out));print(json.dumps({'passed':summary['passed'],'issues':issues}));raise SystemExit(0 if summary['passed'] else 1)
