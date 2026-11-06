import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

home = Path.home()
root = home / 'engram-cache-native-20261106'
window = 'dgx1-1793924574'
image = 'sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b'
deployment = 'f51288df1b30793f7cac0d1dac9bc3cfae4492de'
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
          'started_unix': time.time(), 'commands': [], 'thermal': [],
          'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'fixture_sha256': hashlib.sha256((root / 'gpu-screen.py').read_bytes()).hexdigest()}

def run(argv, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    report['commands'].append({'argv': argv, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if check:
        result.check_returncode()
    return result

def beat(note):
    script = ('import datetime,json,os;from pathlib import Path;'
              'p=Path.home()/"spark-hold.json";x=json.loads(p.read_text());'
              'assert x["window"]==' + repr(window) + ';'
              'assert x["holder"]=="spark-lab: engram-cache native component qualification";'
              'x["heartbeat"]=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ");'
              'x["current"]=' + repr(note) + ';'
              't=p.with_name("engram-heartbeat.tmp");t.write_text(json.dumps(x,indent=2)+"\\n");os.replace(t,p)')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'dgx1', 'python3', '-'],
                            input=script, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('Lost exclusive window: ' + result.stderr)

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
            '--memory', '16g', '--memory-swap', '16g', '--ulimit', 'memlock=-1', '--ipc', 'host',
            '--network', 'none', '-v', str(native / variant) + ':/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/engram_rows:ro',
            '-v', str(root) + ':/input:ro', '-v', str(output) + ':/out',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MAX_JOBS=1',
            '-e', 'XDG_CACHE_HOME=/out/cache', '-e', 'ENGRAM_ROWS_TEST_DIR=/out/direct-tables',
            '--entrypoint', 'python3', image, *command]
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
        run(['sudo', '-n', 'systemctl', 'reset-failed', 'dgx-fan-control.service'])
        run(['sudo', '-n', 'systemctl', 'start', 'dgx-fan-control.service'])
        fan_started = True
    run(['nvidia-smi', '-q', '-x'])
    for variant in ('aligned', 'filter'):
        docker_job('gates-' + variant, variant,
                   ['-m', 'pytest', '-q', '/input/native/' + variant + '/tests/test_deepseek_v41_engram_rows.py'])
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
