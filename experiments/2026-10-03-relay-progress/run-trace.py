from pathlib import Path
import argparse,importlib.machinery,importlib.util,json,subprocess,shlex,time,sys
root=Path(__file__).resolve().parents[2]
l=importlib.machinery.SourceFileLoader('spark3_probe_runner',str(root/'bin/spark3'));s=importlib.util.spec_from_loader(l.name,l);m=importlib.util.module_from_spec(s);l.exec_module(m)
parser=argparse.ArgumentParser(description='Bounded four-rank collective screen')
parser.add_argument('arm')
parser.add_argument('run_id')
parser.add_argument('--benchmark',action='store_true')
parser.add_argument('--counter-samples',action='store_true')
parser.add_argument('--lengths',type=int,nargs='+')
parser.add_argument('--cluster-config',default='experiments/2026-10-03-balanced-policy/selected.json')
parser.add_argument('--output-root',default='.work/relay-progress')
parser.add_argument('--holder',default='relay-progress')
parser.add_argument('--source-profile',default='experiments/2026-10-03-relay-progress/source-only.json')
parser.add_argument('--window',type=int,default=0)
parser.add_argument('--chunk-bytes',type=int,default=0)
parser.add_argument('--trace',action='store_true')
parser.add_argument('--early',action='store_true')
options=parser.parse_args()
kind=options.arm;runid=options.run_id
profile = options.cluster_config
overrides=json.loads((root/'experiments/2026-10-03-collective-policy/arms.json').read_text())[kind]
c,n,_=m.configuration(argparse.Namespace(cluster_config=profile))
_,_,trace_lock=m.configuration(argparse.Namespace(cluster_config=options.source_profile))
trace_inputs=m.build_inputs(trace_lock)
trace_source=m.build_directory(trace_inputs)/'src/b12x'
if m.project_problems(trace_source,trace_inputs['b12x']):raise SystemExit('prepare trace source first')
relative=trace_source.relative_to(root)/'b12x/comm/roce/_roce_proxy.c'
import hashlib
wanted=hashlib.sha256((root/relative).read_bytes()).hexdigest()
source_text=(root/relative).read_text()
for enabled, symbol in ((options.chunk_bytes,'B12X_ROCE_STREAM_CHUNK_BYTES'),
                        (options.window,'B12X_ROCE_STREAM_WINDOW')):
 if enabled and symbol not in source_text: raise SystemExit('selected source does not implement '+symbol)
if options.window and not options.chunk_bytes: raise SystemExit('--window requires --chunk-bytes')
if options.early and not options.trace: raise SystemExit('--early requires --trace')
if options.window and options.trace: raise SystemExit('serial-phase trace attribution is not supported for the shared progress loop')
binding_relative=trace_source.relative_to(root)/'b12x/comm/roce/_proxy.py'
for node in n['nodes']:
 for path in (relative,binding_relative):
  remote=m.repository_path(c)+'/'+str(path)
  got=subprocess.check_output(['ssh','swank@'+node['name'],shlex.join(['sha256sum',remote])],text=True).split()[0]
  if got!=hashlib.sha256((root/path).read_bytes()).hexdigest():raise SystemExit('proxy source mismatch on '+node['name'])
hold=json.loads(subprocess.check_output(['ssh','swank@dgx1','cat ~/spark3-hold.json'],text=True));assert hold['holder']==options.holder
out=root/options.output_root/runid;out.mkdir(parents=True,exist_ok=False)
(out/'invocation.json').write_text(json.dumps(dict(vars(options),proxy_sha256=wanted,source_profile=options.source_profile),indent=2)+'\n')
def hardware_snapshot(suffix):
 code="import pathlib,json,subprocess;root=pathlib.Path('/sys/class/infiniband');d={p.name:{f.name:int(f.read_text()) for f in (p/'ports/1/hw_counters').glob('*')} for p in root.iterdir()};print(json.dumps(d))"
 for node in n['nodes']:
  result=subprocess.run(['ssh','-o','BatchMode=yes','swank@'+node['name'],shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=15,check=True)
  (out/(node['name']+'-hardware-'+suffix+'.json')).write_text(result.stdout)
  extra="import json,subprocess,pathlib;devs=['enp1s0f0np0','enp1s0f1np1','enP2p1s0f0np0','enP2p1s0f1np1'];d={dev:{a.strip():int(b.strip()) for line in subprocess.check_output(['sudo','ethtool','-S',dev],text=True).splitlines() if ':' in line for a,b in [line.split(':',1)] if b.strip().isdigit()} for dev in devs};print(json.dumps({'ports':d,'memory':pathlib.Path('/proc/meminfo').read_text(),'devlink':json.loads(subprocess.check_output(['sudo','devlink','-s','-j','dev','show']))}))"
  result=subprocess.run(['ssh','swank@'+node['name'],shlex.join(['python3','-c',extra])],capture_output=True,text=True,timeout=20,check=True)
  (out/(node['name']+'-ports-'+suffix+'.json')).write_text(result.stdout)
cooling=m.cool_nodes(n,n['nodes'],55,600)
(out/'cooling.json').write_text(json.dumps(cooling,indent=2)+'\n')
hardware_snapshot('before')
processes=[]
for node in n['nodes']:
 cmd=m.collective_probe_command(c,n,node,29581)
 for key,value in overrides.items():
  existing=next((i for i,v in enumerate(cmd) if v.startswith(key+'=')),None)
  if existing is not None: del cmd[existing-1:existing+1]
  if value is not None:
   idx=cmd.index(c['container']['image']);cmd[idx:idx]=['--env',f'{key}={value}']
 for i,v in enumerate(cmd):
  if v.startswith('NCCL_DEBUG_SUBSYS='):cmd[i]='NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING'
 idx=cmd.index(c['container']['image'])
 cmd[idx:idx]=['--volume',m.repository_path(c)+'/'+str(relative)+':/opt/spark3/candidate/b12x/b12x/comm/roce/_roce_proxy.c:ro',
                '--volume',m.repository_path(c)+'/'+str(binding_relative)+':/opt/spark3/candidate/b12x/b12x/comm/roce/_proxy.py:ro',
                '--env','B12X_ROCE_STREAM_CHUNK_BYTES='+str(options.chunk_bytes),
                '--env','B12X_ROCE_STREAM_WINDOW='+str(options.window),
                '--env','B12X_ROCE_TRACE='+('1' if options.trace else '0'),
                '--env','B12X_ROCE_TRACE_EARLY='+('1' if options.early else '0')]
 idx=cmd.index(c['container']['image'])
 cmd[idx:idx]=['--volume','/usr/sbin/ethtool:/usr/local/sbin/ethtool:ro','--volume','/lib/aarch64-linux-gnu/libmnl.so.0:/usr/lib/aarch64-linux-gnu/libmnl.so.0:ro']
 if options.benchmark: cmd.append('--benchmark')
 if options.counter_samples: cmd.append('--counter-samples')
 if options.lengths: cmd += ['--lengths', *map(str,options.lengths)]
 cmd += ['--port-samples','--numerics']
 if kind == 'relay': cmd += ['--expect-paths','2']

 full=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','swank@'+node['name'],shlex.join(cmd)]
 log=open(out/(node['name']+'.txt'),'w')
 errlog=open(out/(node['name']+'-stderr.txt'),'w')
 p=subprocess.Popen(full,stdout=log,stderr=errlog)
 errlog.close()
 processes.append((node,p,log));(out/(node['name']+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
start=time.monotonic();lastbeat=0
while any(p.poll() is None for _,p,_ in processes):
 if time.monotonic()-lastbeat>30:
  beat=f"import json,pathlib,datetime;p=pathlib.Path.home()/'spark3-hold.json';d=json.loads(p.read_text());assert d['holder']=={options.holder!r};d['heartbeat']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(d,indent=2)+'\\n')"
  subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','swank@dgx1',shlex.join(['python3','-c',beat])],timeout=15,check=True)
  lastbeat=time.monotonic()
 if any(p.poll() not in (None,0) for _,p,_ in processes) or time.monotonic()-start>690:
  for node,p,log in processes:
   subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','swank@'+node['name'],'docker rm -f spark3-collective-probe'],capture_output=True,timeout=30)
  break
 time.sleep(1)
results=[]
for node,p,log in processes:
 try:p.wait(timeout=30)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
 log.close();results.append({'node':node['name'],'exit':p.returncode})
 print(node['name'],p.returncode,flush=True)
hardware_snapshot('after')
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')

if any(r['exit'] for r in results): raise SystemExit(1)
