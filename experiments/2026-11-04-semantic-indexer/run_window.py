"""Isolated kernel replay in an owner-authorized idle Spaceport window.

Reuse the measured window's token leases, coordinated start/restore and independent
watchdog. Stop all three C37 ranks once; run bounded kernels only on the head;
restore C37 once. No one-rank serving engine or candidate service is launched.
"""
import argparse
import fcntl
import hashlib
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = HERE.parent/'2026-11-04-clustered-indexer/serving/run_window.py'
spec = importlib.util.spec_from_file_location('coordinated_window', source)
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab

IMAGE = 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024


def thermal():
    return max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))


def review(out, stem):
    plan = (out/(stem+'.txt')).read_text()
    lines = {name: next((x for x in plan.splitlines() if x.startswith(name+':')), None)
             for name in ('live', 'shape')}
    assert all(lines.values()) and ('unchanged' in lines['shape'] or 'new engine' in lines['shape']), lines
    assert '--apply would refuse:' not in plan, plan
    (out/(stem+'-review.json')).write_text(json.dumps(lines, indent=2)+'\n')


def kernel(job, out, token, heartbeat_errors):
    bundle_name, run_name = job.split(':', 1)
    assert bundle_name.startswith('sp8-') and Path(bundle_name).name == bundle_name
    assert Path(run_name).name == run_name
    bundle = HERE/bundle_name
    candidate = json.loads((bundle/'candidate.json').read_text())
    assert candidate['image'] == IMAGE
    destination = out/run_name
    destination.mkdir(exist_ok=False)
    mounted = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p, _ in lab.bundle_mounts(bundle, candidate) if p.is_file()}
    assert window.owns(token) and not heartbeat_errors
    assert not subprocess.check_output(['docker', 'ps', '-q'], text=True).strip()
    assert available() >= 32*2**30 and thermal() < 55
    lab.compile_cache(candidate).mkdir(parents=True, exist_ok=True)
    name = 'semantic-indexer-'+uuid.uuid4().hex[:12]
    command = lab.bundle_command(bundle, candidate, destination)
    command[2:2] = ['--name', name, '--memory', '16g', '--memory-swap', '16g', '--network', 'none']
    (destination/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    began, stamp = time.monotonic(), window.stamp()
    observations = {'minimum_mem_available_bytes': available(), 'maximum_thermal_c': thermal()}
    abort = None
    with (destination/'output.txt').open('w') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                memory, temperature = available(), thermal()
                observations['minimum_mem_available_bytes'] = min(observations['minimum_mem_available_bytes'], memory)
                observations['maximum_thermal_c'] = max(observations['maximum_thermal_c'], temperature)
                if memory < 24*2**30 or heartbeat_errors or time.monotonic()-began > 900:
                    abort = 'memory, lease or per-job duration guard'
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                subprocess.run(['docker', 'stop', '-t', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process.wait(timeout=30)
    output = (destination/'output.txt').read_text()
    verdict = lab.verdict_from_lines(output.splitlines(), candidate)
    verdict.update(exit=process.returncode, image=IMAGE, node=socket.gethostname(),
                   started_at=stamp, ended_at=window.stamp(), seconds=time.monotonic()-began,
                   memory=observations, abort=abort,
                   mounted_inputs_sha256=mounted,
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    verdict['passed'] &= process.returncode == 0 and abort is None
    (destination/'verdict.json').write_text(json.dumps(verdict, indent=2)+'\n')
    for line in reversed(output.splitlines()):
        if line.startswith('RESULT '):
            (destination/'result.json').write_text(json.dumps(json.loads(line[7:]), indent=2)+'\n')
            break
    print(json.dumps({'job': job, 'passed': verdict['passed'], 'seconds': verdict['seconds']}), flush=True)
    if not verdict['passed']:
        raise RuntimeError('kernel failed: '+job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('jobs', nargs='+')
    args = parser.parse_args()
    assert socket.gethostname() == 'spark-0013'
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(['git', 'branch', '-r', '--contains', commit], cwd=ROOT, text=True).strip()
    holds = window.each(window.HOLD_CODE, {'action': 'read'})
    assert not any(any(x.values()) for x in holds.values()), holds
    identity = window.each('''from pathlib import Path
import json,subprocess
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True))[0]
p=Path.home()/'spark-lab/checkouts/clustered-serving'
original=Path.home()/'projects/spark-ds41f'
print(json.dumps({'image':x['Image'],'running':x['State']['Running'],
 'containers':subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines(),
 'checkout':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
 'dirty':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True).strip(),
 'original_checkout':subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip(),
 'original_dirty':subprocess.check_output(['git','status','--porcelain'],cwd=original,text=True).strip()}))
''')
    assert all(x['image'] == window.IMAGE and x['running'] and x['containers'] == ['spark-ds41f']
               and not x['dirty'] and not x['original_dirty'] for x in identity.values()), identity
    assert len({x['checkout'] for x in identity.values()}) == 1
    (out/'preflight-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    window.run(window.spark('restore', 'doctor', '--live'), out, 'preflight-doctor')
    window.run(window.spark('restore', 'cluster', 'start'), out, 'runtime-plan')
    review(out, 'runtime-plan')
    window.run(window.spark('restore', 'cluster', 'sync', '--revision', commit), out, 'sync-plan')
    # A sync plan has no live/shape lines. Read those from the same profile's
    # start plan above before applying the coordinated source-only sync.
    missing = [str(p) for job in args.jobs for p, _ in lab.bundle_mounts(
        HERE/job.split(':', 1)[0], json.loads((HERE/job.split(':', 1)[0]/'candidate.json').read_text()))
               if not p.exists()]
    assert not missing, missing
    if args.preflight_only:
        return
    baseline = None
    for i in range(7):
        metrics = urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=15).read().decode()
        lines = [x for x in metrics.splitlines() if not x.startswith('#') and any(k in x for k in
                 ('num_requests_running{', 'num_requests_waiting{', 'request_success_total{'))]
        for line in lines:
            if 'num_requests_' in line:
                assert float(line.split()[-1]) == 0, line
        successes = [x for x in lines if 'request_success_total{' in x]
        if baseline is None:
            baseline = successes
        assert baseline == successes, 'traffic changed during idle guard'
        (out/f'idle-{i}.json').write_text(json.dumps({'at': window.stamp(), 'metrics': lines}, indent=2)+'\n')
        if i < 6:
            time.sleep(5)
    token = uuid.uuid4().hex
    hold = {'holder': 'semantic-indexer-kernel-window', 'token': token, 'since': window.stamp(),
            'heartbeat': window.stamp(), 'expected_end': None, 'nodes': ['sp5', 'sp6', 'sp7'],
            'rule': 'exclusive isolated replay on sp5; all ranks stopped together; restore recorded C37 once',
            'commit': commit, 'production_config': window.REL+'/restore.json', 'controller_output': str(out)}
    acquired = []
    try:
        for node in window.NODES:
            window.remote(node, window.HOLD_CODE, {'action': 'open', 'hold': hold})
            acquired.append(node)
    except Exception:
        for node in acquired:
            window.remote(node, window.HOLD_CODE, {'action': 'remove', 'token': token})
        raise
    (out/'window.json').write_text(json.dumps(hold, indent=2)+'\n')
    stop, heartbeat_errors = threading.Event(), []
    def beat():
        while not stop.wait(30):
            try:
                window.each(window.HOLD_CODE, {'action': 'refresh', 'token': token, 'heartbeat': window.stamp()})
            except Exception as error:
                heartbeat_errors.append(str(error))
                return
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    def finish():
        stop.set()
        thread.join(timeout=35)
        assert not thread.is_alive(), 'heartbeat writer still active; retaining leases'
    with (out/'watchdog.txt').open('w') as handle:
        subprocess.Popen([sys.executable, str(source), '--out', str(out), '--watchdog', token], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    gpu_lock = None
    try:
        assert window.owns(token) and not heartbeat_errors
        window.run(window.spark('restore', 'cluster', 'sync', '--revision', commit, '--apply'), out, 'sync-apply')
        command = window.spark('restore', 'cluster', 'stop')
        window.run(command, out, 'stop-plan')
        # stop plans do not render shape; runtime-plan records the reviewed shape.
        window.run(command+['--apply'], out, 'stop-apply')
        gpu_lock = open('/tmp/spark-lab-gpu.lock', 'a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in args.jobs:
            requests = window.each(window.HOLD_CODE, {'action': 'read'})
            if any(x['spark-request.json'] or x['spark3-request.json'] for x in requests.values()):
                raise RuntimeError('pending request: restoring group before next job')
            # Untimed correctness work still starts each fresh process cool.
            began = time.monotonic()
            while thermal() >= 55:
                assert window.owns(token) and not heartbeat_errors
                assert time.monotonic()-began < 600, 'cooling timeout'
                time.sleep(5)
            kernel(job, out, token, heartbeat_errors)
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
        try:
            window.restore(token, out, 'final-restore', finish)
        finally:
            finish()


if __name__ == '__main__':
    main()
