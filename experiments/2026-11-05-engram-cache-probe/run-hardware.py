import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import argparse
import signal
import xml.etree.ElementTree as ET

home = Path.home()
parser = argparse.ArgumentParser()
parser.add_argument('--window', required=True)
parser.add_argument('--unit', required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--phase', choices=('gates', 'screen'), required=True)
parser.add_argument('--deployment', required=True)
parser.add_argument('--gates-receipt', type=Path)
args = parser.parse_args()
assert args.run.replace('-', '').replace('_', '').isalnum()
root = home / args.run
window = args.window
image = 'sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b'
deployment = args.deployment
root.mkdir(exist_ok=False)
native = root / 'native'
native.mkdir()
metadata = json.loads((home / 'engram-native-sources.json').read_text())
assert hashlib.sha256((home / 'engram-native-sources.tar.gz').read_bytes()).hexdigest() == metadata['archive_sha256']
with tarfile.open(home / 'engram-native-sources.tar.gz') as archive:
    archive.extractall(native, filter='data')
for name, record in metadata['variants'].items():
    for path, sha in record['files'].items():
        assert hashlib.sha256((native / name / path).read_bytes()).hexdigest() == sha
shutil.copyfile(home / 'engram-gpu-screen.py', root / 'gpu-screen.py')
report = {'deployment': deployment, 'image': image, 'window': window, 'source': metadata,
          'phase': args.phase, 'started_unix': time.time(), 'commands': [], 'thermal': [],
          'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'fixture_sha256': hashlib.sha256((root / 'gpu-screen.py').read_bytes()).hexdigest()}
if args.phase == 'screen':
    assert args.gates_receipt is not None, 'Screen requires the exact-source GPU gate receipt'
    gates = json.loads(args.gates_receipt.read_text())
    assert gates.get('passed') and gates.get('phase') == 'gates'
    assert gates['image'] == image and gates['source'] == metadata
    assert {c['name']: c.get('passed_tests') for c in gates['commands'] if 'passed_tests' in c} == {
        'gates-aligned': 43, 'gates-filter': 43}
    report['gates_receipt_sha256'] = hashlib.sha256(args.gates_receipt.read_bytes()).hexdigest()

ready_path = home / ('.engram-watch-' + args.run + '.json')
ready = json.loads(ready_path.read_text())
assert ready['window'] == window and ready['unit'] == args.unit and ready['mode'] == 'production'
assert time.time() - ready_path.stat().st_mtime < 30
os.kill(ready['pid'], 0)
watch = {'window': window, 'unit': args.unit, 'heartbeat_unix': time.time(),
         'fan_restore_required': False, 'complete': False}
def save_watch():
    watch['heartbeat_unix'] = time.time()
    temporary = root / 'watch.tmp'
    temporary.write_text(json.dumps(watch) + '\n')
    os.replace(temporary, root / 'watch.json')
save_watch()
def cancel(signum, frame):
    raise SystemExit('Received signal ' + str(signum))
signal.signal(signal.SIGTERM, cancel)

def run(argv, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    report['commands'].append({'argv': argv, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if check:
        result.check_returncode()
    return result

def beat(note):
    p = Path.home() / 'spark-hold.json'
    if not p.exists():
        raise RuntimeError('Lost exclusive window: local spark-hold.json absent')
    x = json.loads(p.read_text())
    assert x.get('window') == window
    assert x.get('holder') == 'spark-lab: engram-cache native component qualification'
    assert x.get('pool') == 'dgx4-single' and x.get('nodes') == ['dgx4']
    x['heartbeat'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    x['current'] = note
    t = p.with_name('engram-heartbeat.tmp')
    t.write_text(json.dumps(x, indent=2) + '\n')
    os.replace(t, p)
    save_watch()

def thermal(label):
    zones = {p.parent.name: int(p.read_text()) / 1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    report['thermal'].append({'label': label, 'at_unix': time.time(), 'zones_c': zones, 'available_bytes': available})
    assert available > 24 * 1024**3, 'Host memory floor'
    assert zones and max(zones.values()) < 85, 'Thermal stop'
    return max(zones.values())

def cool(label):
    deadline = time.monotonic() + 600
    while thermal(label) >= 55:
        assert time.monotonic() < deadline, 'Cooling timed out'
        beat('Engram DGX4 cooling before ' + label)
        time.sleep(10)

def docker_job(name, variant, command, timeout=600):
    beat('Engram native component job: ' + name + ' on DGX4; serving unchanged')
    cool(name)
    output = root / name
    output.mkdir()
    container = 'engram-' + name
    argv = ['docker', 'run', '--rm', '--name', container, '--gpus', 'all', '--cpus', '4',
            '--label', 'local.engram.window=' + window, '--label', 'local.engram.run=' + args.run,
            '--memory', '16g', '--memory-swap', '16g', '--ulimit', 'memlock=-1', '--ipc', 'host',
            '--security-opt=seccomp=unconfined', '--cpuset-cpus=5-9,15-19',
            '--network', 'none', '-v', str(native / variant) + ':/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/engram_rows:ro',
            '-v', str(root) + ':/input:ro', '-v', str(output) + ':/out',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MAX_JOBS=1',
            '-e', 'XDG_CACHE_HOME=/out/cache', '-e', 'ENGRAM_ROWS_TEST_DIR=/out/direct-tables',
            '--entrypoint', '/usr/bin/timeout', image, '--kill-after=10', str(timeout), 'python3', *command]
    started = time.monotonic()
    receipt = {'name': name, 'argv': argv, 'started_unix': time.time()}
    report['commands'].append(receipt)
    with (output / 'stdout.log').open('w') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(name)
                beat('Engram native component job: ' + name + ' on DGX4; serving unchanged')
                thermal(name)
                time.sleep(10)
            receipt['exit'] = process.returncode
            assert process.returncode == 0, (name, process.returncode)
            if name.startswith('gates-'):
                cases = list(ET.parse(output / 'junit.xml').getroot().iter('testcase'))
                assert len(cases) == 43 and all(not any(case.find(tag) is not None
                    for tag in ('failure', 'error', 'skipped')) for case in cases)
                receipt['passed_tests'] = len(cases)
        finally:
            if process.poll() is None:
                subprocess.run(['docker', 'stop', '-t', '5', container], capture_output=True, timeout=15)
                process.wait(timeout=15)
            receipt['elapsed_s'] = time.monotonic() - started
            receipt['finished_unix'] = time.time()
            (root / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(name, 'passed', flush=True)

fan_started = False
try:
    beat('Engram native component job setup on DGX4; serving unchanged')
    active = run(['systemctl', 'is-active', '--quiet', 'dgx-fan-control.service'], check=False).returncode == 0
    report['fan_original_active'] = active
    run(['systemctl', 'cat', 'dgx-fan-control.service'])
    if not active:
        watch['fan_restore_required'] = True
        save_watch()
        fan_started = True
        run(['sudo', '-n', 'systemctl', 'reset-failed', 'dgx-fan-control.service'])
        run(['sudo', '-n', 'systemctl', 'start', 'dgx-fan-control.service'])
    run(['nvidia-smi', '-q', '-x'])
    if args.phase == 'gates':
        for variant in ('aligned', 'filter'):
            docker_job('gates-' + variant, variant,
                       ['-m', 'pytest', '-q', '-o', 'cache_dir=/out/pytest-cache',
                        '--junitxml=/out/junit.xml',
                        '/input/native/' + variant + '/tests/test_deepseek_v41_engram_rows.py'])
    else:
        for boot, variants in (('a', ('control', 'aligned', 'filter')), ('b', ('filter', 'aligned', 'control'))):
            for variant in variants:
                command = ['/input/gpu-screen.py', '--native-source',
                           '/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/engram_rows', '--out', '/out/measurements']
                if boot == 'b':
                    command += ['--reverse']
                docker_job('screen-' + boot + '-' + variant, variant, command, timeout=480)
    report['passed'] = True
finally:
    if fan_started:
        run(['sudo', '-n', 'systemctl', 'stop', 'dgx-fan-control.service'])
    report['fan_final_active'] = run(['systemctl', 'is-active', '--quiet', 'dgx-fan-control.service'], check=False).returncode == 0
    report['finished_unix'] = time.time()
    (root / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    if 'fan_original_active' in report:
        assert report['fan_final_active'] == report['fan_original_active'], 'Fan state not restored'
    watch['complete'] = True
    watch['fan_restore_required'] = False
    save_watch()
