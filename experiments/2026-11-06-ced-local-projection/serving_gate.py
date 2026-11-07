"""Bounded full-model correctness on the existing SP5-SP7 serving group."""
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from serving_verdicts import verdict, compare

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
PROFILE = str(EXP.relative_to(ROOT))
IMAGES = {
    'control': 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713',
    'candidate': 'sha256:b1dac2d1ecf9bde9a2412befa40b081745d69d1bc86f76f66bc99e53eb0765bf',
}
SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ControlMaster=auto', '-o', 'ControlPath=/tmp/spark-ssh-%C', '-o', 'ControlPersist=600']
RECORDED_CONTROL = EXP / 'recorded-control.json'
PROBES = [
    ('ced', 'experiments/2026-10-26-spark-ds41f-candidate/ced_cross_shape.py', []),
    ('determinism', 'experiments/2026-10-12-r6g/token_determinism_text.py',
     ['--tokens', '192', '--long-lines', '700', '--wide']),
]
READ = """from pathlib import Path
import json,subprocess,time
h=json.loads((Path.home()/'spark-hold.json').read_text())
zones={p.parent.name:int(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.gr','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
print(json.dumps({'at':time.time(),'hold':h,'thermal_zones_c':zones,'host_available_bytes':available,'gpu_temperature_c':float(gpu[0]),'graphics_mhz':gpu[1].strip()}))
"""
IDENTITY = """from pathlib import Path
import subprocess,json
r=Path('/home/chris/spark-lab/ced-sp567/deployment')
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True,timeout=20))[0]
print(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),
'dirty':subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip(),
'id':x['Id'],'started_at':x['State']['StartedAt'],'running':x['State']['Running'],'restart_count':x['RestartCount'],
'image':x['Image'],'labels':x['Config']['Labels'],'command':x['Config']['Cmd'],
'containers':subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()}))
"""


def state(c, token, floor_gib):
    c.global_scope()
    records = c.recovery.each(READ)
    try:
        for node, row in records.items():
            h = row['hold']
            assert h.get('holder') == c.HOLDER and h.get('token') == token and h.get('nodes') == c.NAMES, node + ': lease changed'
            assert h.get('phase') == 'serving', node + ': phase changed'
            assert row['thermal_zones_c'], node + ': no thermal sensors'
            assert max(row['thermal_zones_c'].values()) <= 78 and row['gpu_temperature_c'] <= 70, node + ': thermal guard'
            assert row['host_available_bytes'] >= floor_gib * 2**30, node + ': memory reserve'
    except AssertionError as error:
        error.observations=records
        raise
    return records


def cool(c, token, out, stem, *, below_c=52.0, sp5_below_c=52.0, settle_polls=2):
    began = time.monotonic()
    observations = []
    settled = 0
    failure = None
    try:
        while True:
            current = state(c, token, 3)
            observations.append(current)
            if all(max(row['thermal_zones_c'].values()) < (sp5_below_c if node in ('10.30.0.13', 'sp5') else below_c) and row['gpu_temperature_c'] < below_c
                   for node, row in current.items()):
                settled += 1
                if settled >= settle_polls:
                    return
            else:
                settled = 0
            assert time.monotonic() - began < 600, 'passive cooling deadline'
            time.sleep(3 if settled else 5)
    except BaseException as error:
        failure={'error':repr(error),'failed_observation':getattr(error,'observations',None)}
        raise
    finally:
        (out / (stem + '-cooling.json')).write_text(json.dumps(
            {'below_c':below_c,'settle_polls':settle_polls,'observations':observations,'failure':failure}, indent=2) + '\n')


def run(c, token, command, out, stem, *, floor_gib, timeout):
    record = {'argv': command, 'started_at': c.stamp(), 'observations': [], 'exit': None,
              'memory_floor_gib': floor_gib, 'max_hottest_c': 70, 'max_gpu_c': 70,
              'timeout_seconds': timeout, 'scope': 'correctness only; stock cooling; no timing qualification'}
    (out / (stem + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
    process = None
    began = time.monotonic()
    try:
        state(c, token, floor_gib)
        with (out / (stem + '.txt')).open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            while process.poll() is None:
                record['observations'].append(state(c, token, floor_gib))
                assert time.monotonic() - began < timeout, stem + ': process deadline'
                time.sleep(5)
            record['exit'] = process.returncode
            assert process.returncode == 0, stem + ': command failed'
    except BaseException as error:
        record['error'] = repr(error)
        record['failed_observation'] = getattr(error,'observations',None)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            record['terminated_exit'] = process.returncode
        record['ended_at'] = c.stamp()
        (out / (stem + '-receipt.json')).write_text(json.dumps(record, indent=2) + '\n')


def identity(c, commit, profile, image):
    records = c.recovery.each(IDENTITY)
    assert all(row['commit'] == commit and not row['dirty'] and row['image'] == image
               and row['running'] and row['restart_count'] == 0
               and row['labels'].get('spark.cluster-config') == profile
               and row['containers'] == ['spark-ds41f'] for row in records.values()), records
    return records


def collect_logs(c, out, before):
    records = {}
    for node in c.recovery.NODES:
        name = c.NAMES[list(c.recovery.NODES).index(node)]
        path = out / (name + '-engine.log')
        row = {'container': before[node]['id'], 'exit': None, 'issues': []}
        try:
            with path.open('xb') as stream:
                result = subprocess.run(['ssh', *SSH_OPTS, 'chris@' + node, 'docker', 'logs', '--tail=100000',
                                         before[node]['id']], stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            row['exit'] = result.returncode
            if result.returncode:
                row['issues'].append('log command failed')
        except Exception as error:
            row['issues'].append(repr(error))
        if path.exists():
            data = path.read_bytes()
            row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if len(data) >= 32 * 2**20:
                row['issues'].append('log size limit reached')
            text = data.decode(errors='replace')
            row['fault_markers'] = [marker for marker in ('Traceback (most recent call last)',
                'CUDA error:', 'illegal memory access', 'device-side assert',
                'EngineCore encountered a fatal error') if marker in text]
            if row['fault_markers']:
                row['issues'].append('engine fault')
        else:
            row['issues'].append('log file missing')
        records[name] = row
    (out / 'engine-logs.json').write_text(json.dumps(records, indent=2) + '\n')
    assert all(not row['issues'] for row in records.values()), records


@contextmanager
def retained_engine_logs(c, out, before):
    """Retain every rank's log before recovery, preserving the original failure."""
    try:
        yield
    except BaseException:
        try:
            collect_logs(c, out, before)
        except BaseException:
            (out / 'engine-log-collection-error.txt').write_text(traceback.format_exc())
        raise
    else:
        collect_logs(c, out, before)


def run_cohort(c, token, out, commit, held, heartbeat_errors):
    arms = {}
    previous = None
    recorded = json.loads(RECORDED_CONTROL.read_text()) if RECORDED_CONTROL.exists() else None
    if recorded is not None:
        (out / 'recorded-control.json').write_text(json.dumps(recorded, indent=2) + '\n')
        for label in ('control-1', 'control-2'):
            arm = out / label
            arm.mkdir()
            entry = recorded['arms'][label]
            (arm / 'recorded-source.json').write_text(json.dumps({'source_window': entry['source_window'], 'note': recorded['note']}, indent=2) + '\n')
            (arm / 'identity-before.json').write_text(json.dumps(entry['identity_before'], indent=2) + '\n')
            arms[label] = {}
            for stem, value in entry['verdicts'].items():
                arms[label][stem] = value
                (arm / (stem + '-verdict.json')).write_text(json.dumps(value, indent=2) + '\n')
                print(json.dumps({'arm': label, 'gate': stem, 'passed': value['passed'], 'issues': value['issues'], 'recorded': True}), flush=True)
    run_labels = ('candidate-1', 'candidate-2') if recorded is not None else ('control-1', 'control-2', 'candidate-1', 'candidate-2')
    for label in run_labels:
        remaining = (dt.datetime.fromisoformat(held['expected_end']) - dt.datetime.now(dt.timezone.utc)).total_seconds()
        assert remaining >= 20 * 60, 'insufficient time for another full arm and restoration'
        assert c.owns(token) and not heartbeat_errors
        c.global_scope()
        assert not any(row['spark-request.json'] or row['spark3-request.json'] for row in c.holds().values()), 'pool requested; yield after completed arm'
        arm = out / label
        arm.mkdir()
        kind = label.split('-')[0]
        profile = PROFILE + '/sp567-serving-' + kind + '.json'
        base = [str(ROOT / 'bin/spark'), '--cluster-config', profile, 'cluster', 'start', '--replace']
        cool(c, token, arm, 'before-start')
        c.logged_run(base, arm, 'start-plan')
        plan = (arm / 'start-plan.txt').read_text()
        assert 'shape:   unchanged' in plan and '--apply would refuse:' not in plan, plan
        run(c, token, base + ['--apply'], arm, 'start', floor_gib=5, timeout=1200)
        before = identity(c, commit, profile, IMAGES[kind])
        if previous is not None:
            assert all(before[node]['id'] != previous[node]['id'] and before[node]['started_at'] != previous[node]['started_at'] for node in before), 'arm reused a container boot'
        previous = before
        (arm / 'identity-before.json').write_text(json.dumps(before, indent=2) + '\n')
        arms[label] = {}
        with retained_engine_logs(c, arm, before):
            for stem, script, arguments in PROBES:
                assert not heartbeat_errors
                cool(c, token, arm, stem)
                command = ['taskset', '-c', '4', 'python3', '-B', str(EXP / 'record_requests.py'),
                           '--out', str(arm / (stem + '-requests')), '--token', token,
                           str(ROOT / script), 'http://127.0.0.1:8000', *arguments]
                run(c, token, command, arm, stem, floor_gib=3, timeout=3600)
                value = verdict(arm / (stem + '.txt'), stem)
                arms[label][stem] = value
                (arm / (stem + '-verdict.json')).write_text(json.dumps(value, indent=2) + '\n')
                print(json.dumps({'arm': label, 'gate': stem, 'passed': value['passed'], 'issues': value['issues']}), flush=True)
                # Negative functional results remain failures. Further correctness-only
                # arms may characterize them; this controller has no timing path.
            after = identity(c, commit, profile, IMAGES[kind])
            assert all(after[node]['id'] == before[node]['id'] for node in before), 'container changed during requests'
            (arm / 'identity-after.json').write_text(json.dumps(after, indent=2) + '\n')
    final = compare(arms)
    final.update(arms=arms, scope='full-model correctness only; any failed gate blocks performance and selection')
    (out / 'functional-verdict.json').write_text(json.dumps(final, indent=2) + '\n')
    assert final['passed'], final['issues']
