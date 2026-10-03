"""Three bounded boots, supervised on dgx1; restore the stopped entry state."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXP = 'experiments/2026-10-03-prose-speculation'
CONFIG = EXP + '/candidate.json'
RAW = ROOT / 'results/private/prose-speculation'
HOLD = Path.home() / 'spark3-hold.json'
HOLDER = 'prose-speculation'
COST = Path('/home/swank/projects/spark3-vllm-ds41f/cache/kkref/dspark-costs/prose-focus-20261003')
IMAGE = 'sha256:aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b'
m = runpy.run_path(str(ROOT / 'bin/spark3'))
nodes = json.loads((ROOT / 'config/nodes-tp3.local.json').read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def beat():
    d = json.loads(HOLD.read_text())
    assert d['holder'] == HOLDER
    d['heartbeat'] = now()
    HOLD.write_text(json.dumps(d, indent=2))


def save(name, data):
    (RAW / name).write_text(json.dumps(data, indent=2) + '\n')


def observe(stage):
    def one(node):
        p = m['run_ssh'](nodes, node, 'sh', '-c',
            'dgx-fan-control status; cat /sys/class/hwmon/hwmon*/fan*_input 2>/dev/null; '
            'nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader')
        assert p.returncode == 0 and 'state=12/12' in p.stdout, (node['name'], p.stdout, p.stderr)
        return {'node': node['name'], 'output': p.stdout}
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(one, nodes['nodes']))
    with (RAW / 'fans.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now(), 'stage': stage, 'nodes': values}) + '\n')


def command(name, argv, timeout=900, fans=True):
    beat()
    receipt = {'argv': argv, 'started': now()}
    deadline = time.monotonic() + timeout
    with (RAW / (name + '.log')).open('x') as log:
        p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while p.poll() is None:
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    beat()
                    if fans:
                        observe(name)
                    if time.monotonic() > deadline:
                        raise TimeoutError(name)
        except BaseException:
            # Keep this supervisor alive until its child has exited. No replacement
            # runner may race a surviving coordinated launch.
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.terminate()
                p.wait(timeout=120)
            raise
        finally:
            receipt.update(finished=now(), exit_code=p.poll())
            save(name + '-command.json', receipt)
    print(name, p.returncode, flush=True)
    return p.returncode


def cluster(action):
    return ['bin/spark3', '--cluster-config', CONFIG, 'cluster', action, '--apply']


def cool(name):
    deadline = time.monotonic() + 600
    while True:
        beat()
        observe(name)
        values = [m['hottest_zone_c'](nodes, n) for n in nodes['nodes']]
        assert all(v is not None for v in values), values
        if max(values) < 45:
            save(name + '.json', {'utc': now(), 'hottest_c': values})
            return
        if time.monotonic() > deadline:
            raise RuntimeError('cooling timeout: ' + repr(values))
        time.sleep(20)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    assert not (RAW / 'session.json').exists(), 'refuse to overwrite a run'
    assert not COST.exists(), 'first boot must profile a new table'
    beat()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    for n in nodes['nodes']:
        p = m['run_ssh'](nodes, n, 'sh', '-c',
            'git -C /home/swank/projects/spark3-ring4-qualification rev-parse HEAD; '
            'git -C /home/swank/projects/spark3-ring4-qualification status --porcelain; '
            'docker ps -q; docker image inspect vllm-ds41f-kkref:04c30fa98e79-r5o --format "{{.Id}}"')
        assert p.returncode == 0 and p.stdout.strip().splitlines() == [revision, IMAGE], (n, p.stdout, p.stderr)
    save('session.json', {'started': now(), 'revision': revision, 'image_id': IMAGE, 'entry_state': 'stopped'})
    controls = {}
    active = False
    restore_errors = []
    digest = None
    try:
        for n in nodes['nodes']:
            controls[n['name']] = m['take_fan_control'](nodes, n)
            save('fan-controls.json', controls)
            assert controls[n['name']]['floor'], n
        for arm in ('fresh', 'pinned-1', 'pinned-2'):
            cool(arm + '-before-start')
            active = True
            assert command(arm + '-start', cluster('start'), 3600) == 0
            paths = list(COST.glob('dspark-costs-*.json'))
            assert len(paths) == 1, paths
            content = paths[0].read_bytes()
            current = hashlib.sha256(content).hexdigest()
            if digest is not None:
                assert current == digest, 'pinned table changed'
            digest = current
            (RAW / (arm + '-costs.json')).write_bytes(content)
            log = subprocess.check_output(['docker', 'logs', 'dsv41-karmic-kraken'], stderr=subprocess.STDOUT, text=True)
            (RAW / (arm + '-boot.log')).write_text(log)
            marker = 'Pinned DSpark cost curves to' if arm == 'fresh' else 'Using pinned DSpark cost curves from'
            assert marker in log and 'Ignoring pinned DSpark' not in log, marker
            cool(arm + '-before-bench')
            argv = ['python3', EXP + '/bench.py', '--cluster-config', CONFIG, 'bench',
                    '--suites', 'quality,decode', '--decode-cases', 'prose,prose-nothink,story,code',
                    '--concurrency', '1', '--min-samples', '12', '--max-samples', '12',
                    '--seed', '0', '--compare', 'none', '--cool-below', '0', '--allow-mismatch',
                    '--output', 'results/private/prose-speculation/' + arm]
            assert command(arm + '-bench', argv, 1500) == 0
            for n in nodes['nodes']:
                p = m['run_ssh'](nodes, n, 'docker', 'logs', 'dsv41-karmic-kraken')
                (RAW / (arm + '-' + n['name'] + '-serving.log')).write_text(p.stdout + p.stderr)
                assert p.returncode == 0
            assert command(arm + '-stop', cluster('stop') + ['--parallel'], 180) == 0
            active = False
    finally:
        if active:
            try:
                if command('cleanup-stop', cluster('stop') + ['--parallel'], 180, fans=False):
                    restore_errors.append('coordinated stop failed')
            except BaseException as e:
                restore_errors.append(repr(e))
        for n in nodes['nodes']:
            if n['name'] in controls:
                try:
                    result = m['restore_fan_control'](nodes, n, controls[n['name']])
                    if str(result).startswith('FAILED'):
                        restore_errors.append(n['name'] + ': ' + result)
                except BaseException as e:
                    restore_errors.append(repr(e))
        save('restoration.json', {'utc': now(), 'errors': restore_errors, 'cost_sha256': digest})
        if not restore_errors:
            beat()
            HOLD.unlink()
        assert not restore_errors, restore_errors


if __name__ == '__main__':
    main()
