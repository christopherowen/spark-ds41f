"""Bounded kernel-only runner; never changes serving state or deployment checkouts."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark
from spark_ds41f import lab

EXP = Path(__file__).resolve().parent
HOLDER = "norm-projection-boundaries"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
stop = threading.Event()
held = False

def head(script):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "dgx1", "python3", "-"], input=script, text=True, capture_output=True, timeout=20)
    if p.returncode: raise RuntimeError(p.stderr or p.stdout)
    return json.loads(p.stdout)

def reservation(action):
    return head("import json, pathlib, datetime\n" +
        "p=pathlib.Path.home()/'spark-hold.json'\n" +
        "req=pathlib.Path.home()/'spark-request.json'\n" +
        "old=pathlib.Path.home()/'spark3-hold.json'\n" +
        f"holder={HOLDER!r}\naction={action!r}\n" +
        "now=datetime.datetime.now(datetime.timezone.utc).isoformat()\n" +
        "record=json.loads(p.read_text()) if p.exists() else None\n" +
        "if action=='claim':\n" +
        "    if record or old.exists(): print(json.dumps({'acquired':False,'holder':record}))\n" +
        "    else:\n" +
        "        data={'holder':holder,'since':now,'expected_end':None,'heartbeat':now,'rule':'Kernel-only dgx4 reservation. No other GPU work or cluster operations. Serving remains unchanged.','serving_untouched':True}\n" +
        "        try:\n" +
        "            with p.open('x') as f: json.dump(data,f,indent=2)\n" +
        "            if req.exists() and json.loads(req.read_text()).get('who')==holder: req.unlink()\n" +
        "            print(json.dumps({'acquired':True}))\n" +
        "        except FileExistsError: print(json.dumps({'acquired':False}))\n" +
        "elif action=='refresh':\n" +
        "    if not record or record.get('holder')!=holder: raise RuntimeError('reservation lost')\n" +
        "    record['heartbeat']=now; temporary=p.with_name('spark-hold-norm-projection.tmp'); temporary.write_text(json.dumps(record,indent=2)); temporary.replace(p)\n" +
        "    print(json.dumps({'requested':req.exists()}))\n" +
        "elif action=='release':\n" +
        "    if record and record.get('holder')==holder: p.unlink()\n" +
        "    print(json.dumps({'released':True}))\n")

def telemetry():
    data={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'SwapFree:', 'SwapTotal:')):
            key, rest=line.split(':',1); data[key]=int(rest.split()[0])*1024
    data['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    data['hottest_c']=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    return data

def heartbeat():
    with (EXP/'runs/telemetry.jsonl').open('a') as f:
        tick=0
        while not stop.wait(5):
            f.write(json.dumps(telemetry())+'\n'); f.flush()
            tick+=1
            if tick%6==0:
                try: reservation('refresh')
                except Exception as e:
                    print('Heartbeat failed:',e,flush=True)
                    return

def main():
    global held
    # Both source and harness must be from a published commit and clean.
    subprocess.run(['git','diff','--exit-code'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--exit-code'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/feat/norm-projection-boundaries'],cwd=ROOT,check=True)
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        claim=reservation('claim')
        if claim.get('acquired'):
            held=True; break
        time.sleep(30)
    if not held: raise RuntimeError('GPU window unavailable after one hour; no GPU work ran')
    thread=threading.Thread(target=heartbeat,daemon=True); thread.start()
    records=[]
    try:
        if subprocess.check_output(['docker','ps','-q'],text=True).strip():
            raise RuntimeError('GPU node has a running container; no work started')
        actual=subprocess.check_output(['docker','image','inspect','spark-ds41f:candidate-38','--format','{{.Id}}'],text=True).strip()
        assert actual==IMAGE,(actual,IMAGE)
        if telemetry()['MemAvailable'] < 32*(1<<30): raise RuntimeError('Less than 32 GiB available')
        nodes=json.loads((Path.home()/'projects/spark-ds41f/config/nodes-ring4.local.json').read_text())
        target=spark.node_by_name(nodes,'dgx4')
        original=lab.bundle_command
        def bounded(*args):
            command=original(*args)
            return command[:2]+['--memory=12g','--memory-swap=12g','--cpus=2','--pids-limit=256']+command[2:]
        lab.bundle_command=bounded
        for run in (1,2):
            if reservation('refresh').get('requested'): break
            cooling=spark.cool_nodes(nodes,[target],55,600)
            fan=spark.start_fan_curve(nodes,[target])
            problems=spark.fan_curve_problems(fan)
            if problems: raise RuntimeError(problems)
            before=telemetry()
            try:
                result=lab.kernel_local(str(EXP/'bundle'),str(EXP/'runs'/f'run-{run}'),False)
            finally:
                spark.stop_fan_curve(nodes,[target],fan)
            records.append({'run':run,'exit':result,'before':before,'after':telemetry(),'cooling':cooling,'fan':fan,'commit':commit})
            (EXP/'runs/run-records.json').write_text(json.dumps(records,indent=2)+'\n')
            if result: raise RuntimeError(f'Kernel run {run} failed; preserved output')
    finally:
        stop.set(); thread.join(timeout=25)
        reservation('release'); held=False

if __name__=='__main__': main()
