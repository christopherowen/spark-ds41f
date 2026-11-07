"""Bounded correctness-only SP5-SP7 window with the published C37 recovery controller."""
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RESTORE_ROOT = Path('/home/chris/spark-lab/checkouts/clustered-serving')
RESTORE_COMMIT = 'e9eab4604cf4e46996d28c0bfba8c9fe75f7d282'
RESTORE_PROFILE = 'experiments/2026-11-04-token-lsa/paged/restore.json'
HOLDER = 'ced-sp567-native-correctness'
NAMES = ['sp5', 'sp6', 'sp7']
SOURCE = '/home/chris/spark-lab/ced-sp567/vllm'
GLOBAL_HOST = None
GLOBAL_OBSERVATION = None
SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ControlMaster=auto', '-o', 'ControlPath=/tmp/spark-ssh-%C', '-o', 'ControlPersist=600']


def output(*command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True, timeout=60).strip()


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_recovery():
    assert output('git', 'rev-parse', 'HEAD', cwd=RESTORE_ROOT) == RESTORE_COMMIT
    assert not output('git', 'status', '--porcelain', cwd=RESTORE_ROOT)
    path = RESTORE_ROOT/'experiments/2026-11-04-clustered-indexer/serving/run_window.py'
    spec = importlib.util.spec_from_file_location('ced_published_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse the published recovery functions with the exact live profile name.
    module.REL = str(Path(RESTORE_PROFILE).parent)
    module.HOLD_CODE = module.HOLD_CODE.replace('import json,os', "import json,os,fcntl\nlock=(Path.home()/'.ced-hold.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)")
    return module


recovery = None


def logged_run(command, out, stem, *, allow_failure=False):
    (out/(stem+'-command.json')).write_text(json.dumps(command, indent=2)+'\n')
    began = stamp()
    with (out/(stem+'.txt')).open('w') as log:
        try:
            result = subprocess.run(command, cwd=RESTORE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=1500 if stem.startswith(('boot-','smoke-')) else 900)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    (out/(stem+'-exit.json')).write_text(json.dumps({'argv':command, 'started_at':began,
                                                   'ended_at':stamp(), 'exit':code}, indent=2)+'\n')
    print(json.dumps({'step':stem,'exit':code}), flush=True)
    if code and not allow_failure:
        raise RuntimeError(stem+' failed; see retained receipt')
    return code


def remote(node, program, data=None):
    code = 'DATA='+repr(data)+'\n'+program
    result = subprocess.run(['ssh', *SSH_OPTS, 'chris@'+node, 'python3 -'], input=code, text=True,
                            capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(node+': '+result.stderr[-2000:])
    return json.loads(result.stdout)


def global_scope():
    if GLOBAL_OBSERVATION is not None:
        observed = json.loads(GLOBAL_OBSERVATION.read_text())
        age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(observed['observed_at'])).total_seconds()
        assert observed.get('source') == 'dgx1:~/spark-hold.json'
        assert observed.get('success') is True and -10 <= age <= 600, 'global hold observation missing, stale or failed'
        assert 0 <= observed['read_duration_seconds'] <= 30
        held = observed['hold']
    else:
        code = "from pathlib import Path; import json; p=Path.home()/'spark-hold.json'; print(p.read_text() if p.exists() else 'null')"
        result = subprocess.run(['ssh', *SSH_OPTS, 'swank@'+GLOBAL_HOST, 'python3 -'], input=code, text=True,
                                capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError('cannot verify the DGX global hold scope')
        held = json.loads(result.stdout)
    if held is not None:
        nodes = held.get('nodes')
        assert isinstance(nodes, list) and nodes and set(nodes) <= {'dgx1', 'dgx2', 'dgx3', 'dgx4'}, held
        assert not set(nodes) & set(NAMES), held
        scope = held.get('resource_scope')
        if scope is not None:
            assert scope.get('kind') == 'nodes', held
            assert scope.get('holder') == held.get('holder') and scope.get('token') == (held.get('window') or held.get('token')), held
            assert scope.get('nodes') == nodes and scope.get('restore', {}).get('nodes') == nodes, held
            assert scope.get('watchdog', {}).get('nodes') == nodes, held
        else:
            assert held.get('holder') and (held.get('window') or held.get('token')) and held.get('production_config'), held
    return held


IDENTITY = """from pathlib import Path
import json,subprocess
root=Path('/home/chris/spark-lab/checkouts/clustered-serving')
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True,timeout=20))[0]
print(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'dirty':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(),
 'image':x['Image'],'running':x['State']['Running'],'profile':x['Config']['Labels'].get('spark.cluster-config'),
 'containers':subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()}))
"""
PHASE = """from pathlib import Path
import json,os,fcntl
lock=(Path.home()/'.ced-hold.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
p=Path.home()/'spark-hold.json';h=json.loads(p.read_text())
assert h.get('token')==DATA['token'] and h.get('holder')=='ced-sp567-native-correctness'
h['phase']=DATA['phase'];q=p.with_name('.ced-hold-'+DATA['token']);q.write_text(json.dumps(h,indent=2)+chr(10));os.replace(q,p)
print('true')
"""
CLEANUP = """import json,subprocess
prefix='ced-sp567-'+DATA[:12]+'-'
names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
records=[]
for name in names:
 if name.startswith(prefix):
  p=subprocess.run(['docker','stop','--time=10',name],capture_output=True,text=True,timeout=25)
  records.append({'name':name,'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
assert all(r['code']==0 for r in records),records
print(json.dumps(records))
"""


def holds():
    return recovery.each(recovery.HOLD_CODE, {'action': 'read'})


def owns(token):
    state = holds()
    return all(value['spark-hold.json'] and value['spark-hold.json'].get('token') == token
               and value['spark-hold.json'].get('holder') == HOLDER
               and value['spark-hold.json'].get('nodes') == NAMES
               and not value['spark3-hold.json'] for value in state.values())


def identity():
    records = recovery.each(IDENTITY)
    assert all(value['commit'] == RESTORE_COMMIT and not value['dirty']
               and value['image'] == recovery.IMAGE and value['running']
               and value['profile'] == RESTORE_PROFILE
               and value['containers'] == ['spark-ds41f'] for value in records.values()), records
    return records


def idle(out):
    baseline = None
    for index in range(7):
        metrics = urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=10).read().decode()
        lines = [line for line in metrics.splitlines() if not line.startswith('#') and any(
            name in line for name in ('num_requests_running{', 'num_requests_waiting{', 'request_success_total{'))]
        assert any('num_requests_running{' in line for line in lines), 'missing running metric'
        assert any('num_requests_waiting{' in line for line in lines), 'missing waiting metric'
        assert all(float(line.split()[-1]) == 0 for line in lines if 'num_requests_' in line), lines
        successes = sorted(line for line in lines if 'request_success_total{' in line)
        assert successes, 'missing completion counter'
        if baseline is None:
            baseline = successes
        assert successes == baseline, 'new serving traffic during idle guard'
        (out/f'idle-{index}.json').write_text(json.dumps({'at': stamp(), 'metrics': lines}, indent=2)+'\n')
        if index < 6:
            time.sleep(5)


def restore(token, out, finish=None):
    with (out/'restore.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'restored.json').exists():
            if finish:
                finish()
            return
        try:
            assert owns(token), 'cohort lease changed; no foreign holder may be overwritten'
            recovery.each(PHASE, {'token': token, 'phase': 'restoring'})
            cleanup = recovery.each(CLEANUP, token)
            (out/'native-cleanup.json').write_text(json.dumps(cleanup, indent=2)+'\n')
            # The serving checkouts were never changed. Refuse checkout drift rather
            # than attempting an unplanned sync during recovery.
            check = recovery.each("""import subprocess,json
p='/home/chris/spark-lab/checkouts/clustered-serving'
print(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
'dirty':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True).strip()}))

""")
            assert all(v['commit']==RESTORE_COMMIT and not v['dirty'] for v in check.values()), check
            # The published function owns coordinated start, live doctor, image/mount
            # verification, heartbeat join before unlink, and token-specific release.
            def before_release():
                final = identity()
                (out/'independent-restoration.json').write_text(json.dumps(final, indent=2)+'\n')
                if finish:
                    finish()
            recovery.restore(token, out, 'restore', finish_heartbeat=before_release)

        except BaseException:
            (out/'restore-failed.txt').write_text(traceback.format_exc())
            if finish:
                finish()
            raise


def watchdog(token, out):
    while not (out/'restored.json').exists() and not (out/'restore-failed.txt').exists():
        time.sleep(15)
        try:
            state = holds()
        except Exception as error:
            with (out/'watchdog-read-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'at':stamp(),'error':repr(error)})+'\n')
            continue
        owned = [v['spark-hold.json'] for v in state.values()
                 if v['spark-hold.json'] and v['spark-hold.json'].get('token') == token]
        if not owned:
            return
        age = max((dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(h['heartbeat'])).total_seconds() for h in owned)
        if age > 900:
            restore(token, out)
            return


def main():
    global recovery, GLOBAL_HOST, GLOBAL_OBSERVATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watchdog')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--job', choices=('native','collective','serving'), default='native')
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument('--global-host', help='verified reachable management address of dgx1')
    route.add_argument('--global-observation', type=Path, help='continuously refreshed authenticated DGX1 observation')
    args = parser.parse_args()
    GLOBAL_HOST = args.global_host
    GLOBAL_OBSERVATION = args.global_observation
    assert socket.gethostname() == 'spark-0013', 'controller must run on SP5'
    recovery = load_recovery()
    recovery.remote = remote
    recovery.run = logged_run
    out = args.out.resolve()
    if args.watchdog:
        (out/'watchdog-ready.json').write_text(json.dumps({'token':args.watchdog,'pid':__import__('os').getpid(),
                                                        'restore_commit':RESTORE_COMMIT,'nodes':NAMES},indent=2)+'\n')
        return watchdog(args.watchdog, out)
    out.mkdir(parents=True, exist_ok=False)
    assert not output('git', 'status', '--porcelain', cwd=ROOT)
    commit = output('git', 'rev-parse', 'HEAD', cwd=ROOT)
    assert 'origin/feat/ced-local-boundary' in output('git', 'branch', '-r', '--contains', commit,
                                                   '--format=%(refname:short)', cwd=ROOT).splitlines()
    (out/'global-scope.json').write_text(json.dumps(global_scope(), indent=2)+'\n')
    state = holds(); assert not any(any(v.values()) for v in state.values()), state
    (out/'preflight-identity.json').write_text(json.dumps(identity(), indent=2)+'\n')
    recovery.run(recovery.spark('restore', 'doctor', '--live'), out, 'preflight-doctor')
    command = recovery.spark('restore', 'cluster', 'stop')
    recovery.run(command, out, 'stop-plan')
    stop_plan=(out/'stop-plan.txt').read_text()
    live=next((line for line in stop_plan.splitlines() if line.startswith('live:')),None)
    assert live and all(name+': stop exact container spark-ds41f' in stop_plan for name in NAMES), stop_plan
    recovery.run(recovery.spark('restore','cluster','start','--replace'),out,'recovery-plan')
    plan=(out/'recovery-plan.txt').read_text()
    shape=next((line for line in plan.splitlines() if line.startswith('shape:')),None)
    assert shape and 'unchanged' in shape and '--apply would refuse:' not in plan, plan
    (out/'plan-review.json').write_text(json.dumps({'stop_live':live,'restore_shape':shape},indent=2)+'\n')
    if args.job=='serving':
        for kind in ('control','candidate'):
            profile=str(EXP.relative_to(ROOT)/('sp567-serving-'+kind+'.json'))
            argv=[str(ROOT/'bin/spark'),'--cluster-config',profile,'cluster','start','--replace']
            recovery.run(argv,out,'serving-'+kind+'-plan')
            text=(out/('serving-'+kind+'-plan.txt')).read_text()
            assert 'shape:   unchanged' in text and '--apply would refuse:' not in text,text
    idle(out)
    if not args.apply:
        print('preflight passed; no ownership or serving state changed', flush=True)
        return
    global_scope()
    state = holds(); assert not any(any(v.values()) for v in state.values()), state
    token = uuid.uuid4().hex
    held = {'holder': HOLDER, 'token': token, 'since': stamp(), 'heartbeat': stamp(),
            'expected_end': (dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=180 if args.job=='serving' else 90 if args.job=='collective' else 45)).isoformat(),
            'nodes': NAMES, 'rule': 'exclusive correctness only; coordinated exact C37 restoration before release',
            'phase': 'preparing', 'job':args.job, 'controller_commit': commit, 'source_commit': '51c1a32cd78cfb0a14087e34dfcc06be30928a0e',
            'restore_commit': RESTORE_COMMIT, 'production_config': RESTORE_PROFILE,
            'controller_output': str(out)}
    acquired = []
    try:
        for node in recovery.NODES:
            remote(node, """from pathlib import Path
import json
home=Path.home()
assert not any((home/name).exists() for name in ('spark-hold.json','spark3-hold.json','spark-request.json','spark3-request.json'))
with (home/'spark-hold.json').open('x') as stream:json.dump(DATA,stream,indent=2)
print('true')
""", held)
            acquired.append(node)
    except BaseException:
        for node in acquired:
            remote(node, recovery.HOLD_CODE, {'action':'remove','token':token})
        raise
    (out/'window.json').write_text(json.dumps(held, indent=2)+'\n')
    stop = threading.Event(); errors = []
    def heartbeat():
        while not stop.wait(30):
            try:
                recovery.each(recovery.HOLD_CODE, {'action':'refresh','token':token,'heartbeat':stamp()})
            except Exception as error:
                errors.append(repr(error)); return
    writer = threading.Thread(target=heartbeat, daemon=True)
    def finish():
        stop.set()
        if writer.ident is not None:
            writer.join(timeout=50)
        assert not writer.is_alive(), 'heartbeat writer still active; retain holds'
    try:
        writer.start()
        route_args = ['--global-observation', str(GLOBAL_OBSERVATION)] if GLOBAL_OBSERVATION else ['--global-host', GLOBAL_HOST]
        with (out/'watchdog.txt').open('w') as log:
            guard = subprocess.Popen([sys.executable, __file__, '--out', str(out), '--watchdog', token]+route_args,
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        (out/'watchdog.json').write_text(json.dumps({'pid':guard.pid,'token':token,'targets':NAMES,
                                                   'restore_commit':RESTORE_COMMIT}, indent=2)+'\n')
        deadline=time.monotonic()+15
        while not (out/'watchdog-ready.json').exists():
            assert guard.poll() is None and time.monotonic()<deadline, 'watchdog failed to arm'
            time.sleep(0.25)
        ready=json.loads((out/'watchdog-ready.json').read_text())
        assert ready['token']==token and ready['nodes']==NAMES
        scope={'kind':'nodes','nodes':NAMES,'holder':HOLDER,'token':token,
               'controller':{'commit':commit,'path':str(Path(__file__).resolve()),'pid':__import__('os').getpid()},
               'restore':{'commit':RESTORE_COMMIT,'profile':RESTORE_PROFILE,'image':recovery.IMAGE,'nodes':NAMES},
               'watchdog':{'pid':guard.pid,'nodes':NAMES,'stale_seconds':900}}
        recovery.each("""from pathlib import Path
import json,os,fcntl
lock=(Path.home()/'.ced-hold.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
p=Path.home()/'spark-hold.json';h=json.loads(p.read_text());assert h.get('token')==DATA['token']
h['resource_scope']=DATA;q=p.with_name('.ced-scope-'+DATA['token']);q.write_text(json.dumps(h,indent=2)+'\\n');os.replace(q,p)
print('true')
""",scope)
        (out/'resource-scope.json').write_text(json.dumps(scope,indent=2)+'\n')
        assert owns(token) and guard.poll() is None and not errors
        global_scope()
        # Execute the exact phase writer while service is still healthy.
        recovery.each(PHASE, {'token':token,'phase':'preparing'})
        recovery.run(command+['--apply'], out, 'stop-apply')
        def batch(boot,job):
            remaining=(dt.datetime.fromisoformat(held['expected_end'])-dt.datetime.now(dt.timezone.utc)).total_seconds()
            if remaining < 35*60:
                raise RuntimeError('insufficient time for another bounded batch plus restoration')
            assert owns(token) and not errors
            global_scope()
            state=holds()
            if any(v['spark-request.json'] or v['spark3-request.json'] for v in state.values()):
                raise RuntimeError('cohort requested; yield after completed batch')
            def one(pair):
                rank,node=pair
                argv=['ssh',*SSH_OPTS,'chris@'+node,
                      'python3',str(EXP/'native_worker.py'),'--token',token,'--boot',boot,'--source',SOURCE,
                      '--job',job,'--rank',str(rank)]
                return recovery.run(argv,out,boot+'-'+node.replace('.','-'),allow_failure=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                results=list(pool.map(one,enumerate(recovery.NODES)))
            assert results==[0,0,0], results
        if args.job=='serving':
            recovery.each(PHASE, {'token':token,'phase':'serving'})
            from serving_gate import run_cohort
            run_cohort(sys.modules[__name__],token,out,commit,held,errors)
        else:
            if args.job=='collective':
                recovery.each(PHASE, {'token':token,'phase':'copying'})
                command=['ssh',*SSH_OPTS,'chris@10.30.0.16',
                         'systemd-run','--user','--scope','-p','MemoryMax=4G','-p','CPUQuota=100%',
                         'taskset','-c','4','flock','-n','/tmp/spark-cpu-analysis.lock','python3','-B',
                         '/home/chris/spark-lab/ced-local-boundary/deployment/experiments/2026-11-06-ced-local-projection/distribute_sp567.py',
                         '--token',token,'--commit',commit,'--apply']
                recovery.run(command,out,'image-distribution')
            recovery.each(PHASE, {'token':token,'phase':'native'})
            if args.job=='collective':batch('smoke','smoke')
            for boot in ('boot-1','boot-2'):batch(boot,args.job)
        completion={'at':stamp(),'boots':2,'nodes':NAMES,
                    'scope':args.job+' correctness only; no performance claim'}
        if args.job=='serving':completion.update(images=['control','candidate'],boots_per_image=2)
        filename='functional-complete.json' if args.job=='serving' else 'native-complete.json'
        (out/filename).write_text(json.dumps(completion,indent=2)+'\n')
    except BaseException:
        (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        restore(token,out,finish)


if __name__ == '__main__':
    main()
