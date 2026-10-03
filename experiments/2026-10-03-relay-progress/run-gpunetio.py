"""Bounded adjacent-pair compatibility/latency test; no host package changes."""
import argparse
import datetime
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('run_id')
p.add_argument('--server',type=int,default=0,choices=range(4))
p.add_argument('--lane',type=int,default=0,choices=(0,1))
p.add_argument('--handler',type=int,default=1,choices=(1,))
p.add_argument('--source',choices=('spark','debug','syndrome','system'),default='system')
p.add_argument('--iterations',type=int,default=512)
a=p.parse_args()
if a.iterations <= 0: p.error('--iterations must be positive')
l=importlib.machinery.SourceFileLoader('relay_run',str(ROOT/'bin/spark3'))
s=importlib.util.spec_from_loader(l.name,l);m=importlib.util.module_from_spec(s);l.exec_module(m)
c,n,_=m.configuration(argparse.Namespace(cluster_config='experiments/2026-10-03-balanced-policy/selected.json'))
hold=json.loads(subprocess.check_output(['ssh','swank@dgx1','cat ~/spark3-hold.json'],text=True));assert hold['holder']=='relay-progress'
def heartbeat():
 code="import json,pathlib,datetime;p=pathlib.Path.home()/'spark3-hold.json';d=json.loads(p.read_text());assert d['holder']=='relay-progress';d['heartbeat']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(d,indent=2)+'\\n')"
 subprocess.run(['ssh','swank@dgx1',shlex.join(['python3','-c',code])],check=True,timeout=15)
heartbeat()
out=ROOT/'.work/relay-progress'/a.run_id;out.mkdir(parents=True,exist_ok=False)
(out/'invocation.json').write_text(json.dumps(vars(a),indent=2)+'\n')
(out/'cooling.json').write_text(json.dumps(m.cool_nodes(n,n['nodes'],55,600),indent=2)+'\n')
heartbeat()
server=n['nodes'][a.server];client=n['nodes'][(a.server+1)%4]
children=[]
try:
 for role,node,peer in [('server',server,client),('client',client,server)]:
  if role=='client':
   deadline=time.monotonic()+20
   while 'Listening for incoming connections' not in (out/'server.txt').read_text() and children[0][0].poll() is None and time.monotonic()<deadline:
    time.sleep(.1)
   if 'Listening for incoming connections' not in (out/'server.txt').read_text():break
  source=m.repository_path(c)+'/.work/gpunetio-586453728bca'
  hca=node['roce_peer_hcas'][str(peer['rank'])][a.lane]
  command=['env','DOCA_GPUNETIO_LOG=6','LD_LIBRARY_PATH='+source+'/lib:/usr/local/cuda-13.0/lib64','timeout','--signal=TERM','--kill-after=3','40','stdbuf','-oL',source+'/examples/gpunetio_verbs_write_lat/gpunetio_verbs_write_lat','-g','000f:01:00.0','-d',hca,'-l','3','-p',str(a.handler),'-i',str(a.iterations)]
  if a.source in ('debug','syndrome','system'):command.insert(1,'GPUNETIO_TRACE_PROGRESS=1')
  if role=='client':command+=['-c',server['management_ip']]
  identity=subprocess.check_output(['ssh','swank@'+node['name'],shlex.join(['git','-C',source,'rev-parse','HEAD'])],text=True).strip()
  if identity!=json.loads((Path(__file__).parent/'gpunetio-sources.json').read_text())[a.source]:raise RuntimeError('source identity mismatch')
  (out/(role+'-command.json')).write_text(json.dumps({'node':node['name'],'source_head':identity,'argv':command},indent=2)+'\n')
  f=(out/(role+'.txt')).open('w')
  proc=subprocess.Popen(['ssh','-o','BatchMode=yes','swank@'+node['name'],shlex.join(command)],stdout=f,stderr=subprocess.STDOUT)
  children.append((proc,f,role))
 for proc,f,role in children:
  try:proc.wait(timeout=50)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
finally:
 for proc,f,role in children:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
  f.close()
 result=[{'role':role,'exit':proc.returncode} for proc,f,role in children]
 (out/'results.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':result},indent=2)+'\n')
 print(json.dumps(result))
if len(children)!=2 or any(x[0].returncode for x in children):raise SystemExit(1)
