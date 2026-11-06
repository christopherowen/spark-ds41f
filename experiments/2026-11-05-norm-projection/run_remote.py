"""Bounded kernel-only runner; never changes serving state or deployment checkouts."""
import atexit
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark
from spark_ds41f import lab
from spark_ds41f.bench import THERMAL_FIELDS, parse_thermal, BOARD_CEILING_C, THERMAL_HEADROOM_FLOOR_C

EXP = Path(__file__).resolve().parent
HOLDER = "norm-projection-boundaries"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
stop = threading.Event()
held = False
CONTAINER = "norm-projection-screen"

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
        "requester=json.loads(req.read_text()) if req.exists() else None\n" +
        "if action=='claim':\n" +
        "    if record or old.exists() or (requester and requester.get('who')!=holder): print(json.dumps({'acquired':False,'holder':record,'request':requester}))\n" +
        "    else:\n" +
        "        data={'holder':holder,'since':now,'expected_end':None,'heartbeat':now,'rule':'Kernel-only dgx4 reservation. No other GPU work or cluster operations. Serving remains unchanged.','serving_untouched':True}\n" +
        "        try:\n" +
        "            with p.open('x') as f: json.dump(data,f,indent=2)\n" +
        "            if req.exists() and json.loads(req.read_text()).get('who')==holder: req.unlink()\n" +
        "            print(json.dumps({'acquired':True,'since':now}))\n" +
        "        except FileExistsError: print(json.dumps({'acquired':False}))\n" +
        "elif action=='request':\n" +
        "    if requester: print(json.dumps({'queued':False,'request':requester}))\n" +
        "    else:\n" +
        "        data={'who':holder,'why':'Resume published kernel-only DGX4 norm/projection boundary screen after the active job; no serving changes in this job','how_long':'30 minutes for two fresh process runs; yield after the active run on request','since':now,'branch':'feat/norm-projection-boundaries'}\n" +
        "        try:\n" +
        "            with req.open('x') as f: json.dump(data,f,indent=2)\n" +
        "            print(json.dumps({'queued':True}))\n" +
        "        except FileExistsError: print(json.dumps({'queued':False}))\n" +
        "elif action=='cancel_request':\n" +
        "    if requester and requester.get('who')==holder: req.unlink()\n" +
        "    print(json.dumps({'cancelled':True}))\n" +
        "elif action=='refresh':\n" +
        "    if not record or record.get('holder')!=holder: raise RuntimeError('reservation lost')\n" +
        "    record['heartbeat']=now; temporary=p.with_name('spark-hold-norm-projection.tmp'); temporary.write_text(json.dumps(record,indent=2)); temporary.replace(p)\n" +
        "    print(json.dumps({'requested':req.exists()}))\n" +
        "elif action=='release':\n" +
        "    if record and record.get('holder')==holder: p.unlink()\n" +
        "    print(json.dumps({'released':True}))\n")

def stop_owned_container():
    inspected=subprocess.run(['docker','inspect',CONTAINER,'--format','{{index .Config.Labels "spark.experiment"}} {{.State.Running}}'],text=True,capture_output=True,timeout=15)
    if inspected.returncode:
        if "no such" in inspected.stderr.lower(): return
        raise RuntimeError(inspected.stderr or "Cannot verify owned container state")
    label, running=inspected.stdout.strip().split()
    if label != HOLDER: raise RuntimeError('Container name is owned by another task; leaving it untouched')
    if running == 'true':
        subprocess.run(['docker','stop','--time','10',CONTAINER],check=True,timeout=25)

def cleanup():
    global held
    if not held:
        reservation('cancel_request')
        return
    stop_owned_container()
    stop.set()
    reservation('release')
    held=False

if '--watchdog' not in sys.argv:
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit('Runner terminated; cleaning up its own kernel container'))

def watchdog(since):
    while True:
        try:
            record=head("import json,pathlib\np=pathlib.Path.home()/'spark-hold.json'\nprint(p.read_text() if p.exists() else 'null')\n")
        except Exception as error:
            print('Watchdog reservation read failed; retaining the hold:',error,flush=True)
            time.sleep(30)
            continue
        if not record or record.get('holder')!=HOLDER or record.get('since')!=since:
            return
        if not record.get('serving_untouched'):
            raise RuntimeError('Kernel watchdog refuses a serving window')
        stamp=datetime.datetime.fromisoformat(record['heartbeat'].replace('Z','+00:00'))
        age=(datetime.datetime.now(datetime.timezone.utc)-stamp).total_seconds()
        if age>900:
            print('Stale kernel-window heartbeat; stopping only the owned screen',flush=True)
            subprocess.run(['systemctl','--user','stop','norm-projection-screen.service'],check=True,timeout=60)
            stop_owned_container()
            reservation('release')
            return
        time.sleep(30)


def telemetry(gpu=False):
    data={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'SwapFree:', 'SwapTotal:')):
            key, rest=line.split(':',1); data[key]=int(rest.split()[0])*1024
    data['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    data['hottest_c']=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    if gpu:
        query=subprocess.run(['nvidia-smi','--query-gpu='+','.join(THERMAL_FIELDS),'--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
        if query.returncode: raise RuntimeError(query.stderr or 'GPU telemetry failed')
        data['gpu']=parse_thermal(query.stdout.strip()+','+str(data['hottest_c']*1000))
    return data

def heartbeat():
    with (EXP/'runs/telemetry.jsonl').open('a') as f:
        tick=0
        while not stop.wait(5):
            tick+=1
            try:
                sample=telemetry(gpu=tick%3==0)
                f.write(json.dumps(sample)+'\n'); f.flush()
                headroom=sample.get('gpu',{}).get('headroom_c')
                if sample['MemAvailable'] < 16*(1<<30) or sample['hottest_c'] >= BOARD_CEILING_C or (headroom is not None and headroom < THERMAL_HEADROOM_FLOOR_C):
                    raise RuntimeError('Memory or thermal limit reached')
                if tick%6==0: reservation('refresh')
            except Exception as e:
                print('Monitoring failed:',e,flush=True)
                os.kill(os.getpid(),signal.SIGTERM)
                return

def main():
    global held
    # Both source and harness must be from a published commit and clean.
    subprocess.run(['git','diff','--exit-code'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--exit-code'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/feat/norm-projection-boundaries'],cwd=ROOT,check=True)
    records_path=EXP/'runs/run-records.json'
    records=json.loads(records_path.read_text()) if records_path.exists() else []
    if any(record['commit']!=commit or record['exit'] for record in records):
        raise RuntimeError('Existing receipts need review; refusing to overwrite a different or failed run')
    completed={record['run'] for record in records}
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        claim=reservation('claim')
        if claim.get('acquired'):
            held=True; break
        reservation('request')
        time.sleep(30)
    if not held: raise RuntimeError('GPU window unavailable after one hour; no GPU work ran')
    subprocess.run(['systemd-run','--user','--collect',f'--unit=norm-projection-watchdog-{os.getpid()}',
                    '--property=CPUQuota=10%','--property=MemoryMax=256M','--property=TasksMax=64',
                    '--property=StandardOutput=append:'+str(EXP/'runs/watchdog.txt'),'--property=StandardError=inherit',
                    sys.executable,str(Path(__file__).resolve()),'--watchdog',claim['since']],check=True)
    thread=threading.Thread(target=heartbeat,daemon=True); thread.start()
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
            return command[:2]+['--name',CONTAINER,'--label',f'spark.experiment={HOLDER}','--memory=12g','--memory-swap=12g','--cpus=2','--pids-limit=256']+command[2:]
        lab.bundle_command=bounded
        for run in (1,2):
            if run in completed: continue
            if (EXP/'runs'/f'run-{run}').exists(): raise RuntimeError('Existing output directory; refusing to overwrite it')
            if reservation('refresh').get('requested'): break
            cooling=spark.cool_nodes(nodes,[target],55,600)
            fan=spark.start_fan_curve(nodes,[target])
            problems=spark.fan_curve_problems(fan)
            if problems: raise RuntimeError(problems)
            before=telemetry(gpu=True)
            try:
                result=lab.kernel_local(str(EXP/'bundle'),str(EXP/'runs'/f'run-{run}'),False)
            finally:
                spark.stop_fan_curve(nodes,[target],fan)
            records.append({'run':run,'exit':result,'before':before,'after':telemetry(gpu=True),'cooling':cooling,'fan':fan,'commit':commit})
            (EXP/'runs/run-records.json').write_text(json.dumps(records,indent=2)+'\n')
            if result: raise RuntimeError(f'Kernel run {run} failed; preserved output')
    finally:
        stop_owned_container()
        stop.set(); thread.join(timeout=25)
        reservation('release'); held=False

if __name__=='__main__':
    if '--watchdog' in sys.argv: watchdog(sys.argv[2])
    else: main()
