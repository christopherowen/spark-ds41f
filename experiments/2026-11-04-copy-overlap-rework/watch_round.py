#!/usr/bin/env python3
"""Observe compact C41 receipts only; this program never controls a server."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

PROGRAM = r'''
import datetime,json,os,pathlib,time
root=pathlib.Path('/home/swank/projects/spark-ds41f')
setup=pathlib.Path('/home/swank/tl-logs/copy-overlap-rework/setup')
out=root/'results/private/lab/candidate-41-round-attempt-2'
def read(path):
 if not path.exists():return None
 for attempt in range(3):
  try:return json.loads(path.read_text())
  except json.JSONDecodeError:
   if attempt==2:raise
   time.sleep(0.02)
c=read(out/'receipt.json') or {}
r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller':{k:c.get(k) for k in ('started','window','commit','phase','resumed_from','completed','restored','ended','error','restore_error','management_loss_latched','management_unreachable')},'steps':[{k:s.get(k) for k in ('log','exit','started','ended')} for s in c.get('steps',[])],'alive':{},'process_states':{}}
launch=read(setup/'launch-attempt-2.json')
for key in ('controller_pid','supervisor_pid'):
 pid=launch[key]
 try:
  os.kill(pid,0);state=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]
  r['alive'][str(pid)]=state!='Z';r['process_states'][str(pid)]=state
 except (ProcessLookupError,FileNotFoundError):r['alive'][str(pid)]=False;r['process_states'][str(pid)]='absent'
for key,name in [('hold','spark-hold.json'),('request','spark-request.json'),('legacy_request','spark3-request.json')]:
 r[key]=read(pathlib.Path.home()/name)
r['actual_exit']=read(setup/'exit-attempt-2.json')
h=r.get('hold') or {};hb=h.get('heartbeat')
r['heartbeat_stale_120s']=bool(hb and (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(hb.replace('Z','+00:00'))).total_seconds()>120)
f=read(out/'functional-exits.json')
if f:r['functional_exits']={'passed':f.get('passed'),'ordered_count':len(f.get('ordered_exits',[])),'nonzero':[x for x in f.get('ordered_exits',[]) if x[-1]!='0']}
r['suites']=[]
for p in sorted((root/'results/private/suites').glob('candidate-41-release-tp3*/candidate-41*/suite.json')):
 s=read(p);q={'path':str(p.relative_to(root)),'finished':s.get('finished_utc'),'steps':[{k:x.get(k) for k in ('id','exit')} for x in s.get('steps',[])]}
 x=read(p.parent/'conditions.json')
 if x:q['guards']={'halted':x.get('halted'),'fan_stop_failures':x.get('fan_stop_failures')}
 r['suites'].append(q)
print(json.dumps(r))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compile(PROGRAM, "remote-compact-receipt-read", "exec")
    last = None
    with args.output.open("x") as log:
        while True:
            try:
                p = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx1", "python3", "-"],
                    input=PROGRAM, capture_output=True, text=True, timeout=25,
                )
                if p.returncode:
                    raise RuntimeError(f"read exit {p.returncode}: {p.stderr}")
                row = json.loads(p.stdout)
            except Exception as error:
                row = {"time": datetime.now(timezone.utc).isoformat(),
                       "read_error": str(error), "no_node_actions": True}
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                return 3
            hold = row.get("hold") or {}
            key = json.dumps({k: v for k, v in row.items() if k not in ("time", "hold")}
                             | {k: hold.get(k) for k in ("holder", "window", "current")}, sort_keys=True)
            if key != last:
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                last = key
            if row.get("actual_exit") and not any(row["alive"].values()):
                return 0
            if "hardware-review" in hold.get("holder", ""):
                return 3
            time.sleep(45)


if __name__ == "__main__":
    raise SystemExit(main())
