"""Wrap the existing four-rank probe with read-only host telemetry."""
from pathlib import Path
import subprocess,sys,shlex,json,time
ROOT=Path(__file__).resolve().parents[2]
profile,runid=sys.argv[1:]
output=ROOT/'.work/ring4-bidirectional'/runid
output.mkdir(exist_ok=False)
code=r'''
import pathlib,subprocess,json,datetime,time,select,sys,os
P=pathlib.Path
hz=os.sysconf('SC_CLK_TCK')
def read(p):
 try:return p.read_text().strip()
 except OSError:return None
while not select.select([sys.stdin],[],[],0)[0]:
 start=time.monotonic()
 d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hz':hz,'cpu':read(P('/proc/stat')).splitlines()[0]}
 d['zones']={p.name:read(p/'temp') for p in P('/sys/class/thermal').glob('thermal_zone*')}
 d['fans']={str(p):read(p) for p in P('/sys/class/hwmon').glob('hwmon*/fan*_input')}
 d['threads']={}
 for proc in P('/proc').glob('[0-9]*'):
  cmd=read(proc/'cmdline')
  if not cmd or not any(x.endswith('/probe_collectives.py') for x in cmd.split('\0')):continue
  for stat in (proc/'task').glob('*/stat'):
   s=read(stat)
   if not s:continue
   v=s.rsplit(')',1)[1].split();d['threads'][proc.name+':'+stat.parent.name]={'ticks':int(v[11])+int(v[12]),'state':v[0]}
 r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
 d['gpu']=r.stdout.strip();d['gpu_error']=r.stderr.strip() if r.returncode else None
 print(json.dumps(d),flush=True)
 select.select([sys.stdin],[],[],max(0,2-(time.monotonic()-start)))
'''
children=[]
try:
 for host in ('dgx1','dgx2','dgx3','dgx4'):
  f=(output/(host+'-telemetry.jsonl')).open('w')
  p=subprocess.Popen(['ssh','-o','BatchMode=yes','swank@'+host,shlex.join(['python3','-u','-c',code])],stdin=subprocess.PIPE,stdout=f,stderr=f)
  children.append((p,f))
 cmd=[sys.executable,'experiments/2026-10-03-collective-policy/run.py','relay',runid,'--benchmark','--counter-samples','--cluster-config',profile,'--output-root','.work/ring4-bidirectional/probes','--holder','ring4-bidirectional']
 (output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 subprocess.run(cmd,cwd=ROOT,check=True)
finally:
 for p,f in children:p.stdin.close()
 for p,f in children:
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
  f.close()
