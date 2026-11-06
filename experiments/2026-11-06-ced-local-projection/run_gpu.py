"""Bounded kernel-only qualification on an exclusively reserved idle GPU host."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--window', required=True)
p.add_argument('--boot', required=True)
p.add_argument('--mode', choices=['gates', 'screen'], required=True)
p.add_argument('--schedules')
a = p.parse_args()
exp = Path(__file__).resolve().parent
root = exp.parents[1]
sys.path.insert(0, str(root))
from cooling import helpers

source = a.source.resolve()
identity = json.loads((exp / 'identity.json').read_text())
image = 'sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11'
control = 'swank@dgx1'


def output(*cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def hold(beat=False):
    code = '''import datetime,json,os,sys
from pathlib import Path
p=Path('/home/swank/spark-hold.json');d=json.loads(p.read_text())
assert d['window']==sys.argv[1] and 'ced-local-boundary' in d['holder'], 'window ownership lost'
if sys.argv[2]=='beat':
 d['heartbeat']=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
 q=p.with_name('.ced-heartbeat.json');q.write_text(json.dumps(d,indent=2)+'\\n');os.replace(q,p)
print(json.dumps(d))
'''
    import shlex
    return json.loads(output('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                             control, 'python3', '-c', shlex.quote(code),
                             shlex.quote(a.window), 'beat' if beat else 'read'))


def thermal_zones():
    zones = {path.parent.name: int(path.read_text().strip()) / 1000
             for path in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    if not zones:
        raise RuntimeError('cannot read the hottest thermal zone')
    return zones


def fan_active():
    return subprocess.run(['systemctl', 'is-active', '--quiet', 'dgx-fan-control.service'],
                          check=False, timeout=15).returncode == 0


def telemetry():
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    fields = output('nvidia-smi', '--query-gpu=uuid,temperature.gpu,clocks.gr,clocks.mem,memory.used,power.draw',
                    '--format=csv,noheader,nounits').split(',')
    assert len(fields) == 6, fields
    def number(value):
        try:
            return float(value)
        except ValueError:
            return None  # GB10 does not expose every discrete-GPU telemetry field.
    return {'thermal_zones_c': thermal_zones(), 'performance_fan_active': fan_active(), 'raw_gpu_fields': [field.strip() for field in fields], 'time': time.time(), 'host_available_bytes': available, 'gpu_uuid': fields[0].strip(),
            'temperature_c': float(fields[1]), 'graphics_mhz': number(fields[2]),
            'memory_mhz': number(fields[3]), 'gpu_used_mib': number(fields[4]), 'power_w': number(fields[5])}


window = hold()
assert not output('git', 'status', '--porcelain', cwd=root), 'deployment checkout must be clean'
assert not output('git', 'status', '--porcelain', cwd=source), 'source checkout must be clean'
assert output('git', 'rev-parse', 'HEAD', cwd=source) == identity['candidate']['patch_head']
assert output('git', 'rev-parse', 'HEAD^{tree}', cwd=source) == identity['candidate']['expected_tree']
assert not output('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'), 'GPU is busy'
first = telemetry()
assert first['host_available_bytes'] >= 48 * 2**30 and first['temperature_c'] <= 70, first
images = json.loads(output('docker', 'image', 'inspect', image))
labels = images[0]['Config']['Labels']
assert labels['local.spark-ds41f.vllm.tree'] == identity['unreleased_control']['tree']
for project in ('tilelang', 'tile_kernels', 'b12x', 'nccl', 'sparknet'):
    expected = json.loads((exp / 'source.json').read_text())[project]['expected_tree']
    assert labels[f'local.spark-ds41f.{project}.tree'] == expected, project
changed = output('git', 'diff', '--name-only', identity['unreleased_control']['head'], 'HEAD', cwd=source).splitlines()
code_files = [name for name in changed if name.startswith('vllm/')]
assert len(code_files) == 6 and all(name.endswith('.py') for name in code_files), code_files
stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
name = f'ced-{a.mode}-{stamp.lower()}'
result = root / 'results/private/ced-local-projection' / f'{a.boot}-{a.mode}-{stamp}'
result.mkdir(parents=True)
cache = root / 'results/private/ced-local-projection/cache'
cache.mkdir(parents=True, exist_ok=True)
command = ['docker', 'run', '--rm', '--name', name, '--gpus', 'all', '--cpus=1', '--memory=16g',
           '--memory-swap=16g', '--pids-limit=256', '--network=none', '--shm-size=1g',
           '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MAX_JOBS=1',
           '-e', 'NVCC_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'VLLM_CACHE_ROOT=/root/.cache/vllm',
           '-v', f'{source}:/src:ro', '-v', f'{exp}:/check:ro', '-v', f'{cache}:/root/.cache:rw']
for file in code_files:
    command += ['-v', f'{source / file}:/opt/spark-ds41f/candidate/vllm/{file}:ro']
command += ['--entrypoint', 'python3', image, '/check/gpu.py' if a.mode == 'gates' else '/check/screen_projection.py']
if a.mode == 'screen':
    command += ['--boot', a.boot]
    if a.schedules:
        command += ['--schedules', a.schedules]
receipt = {'window': window, 'host': socket.gethostname(), 'source': identity['candidate'],
           'deployment_commit': output('git', 'rev-parse', 'HEAD', cwd=root), 'image': image,
           'image_labels': labels, 'image_config_sha256': hashlib.sha256(json.dumps(images[0]['Config'], sort_keys=True).encode()).hexdigest(),
           'image_layers': images[0]['RootFS']['Layers'], 'command': command,
           'file_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in code_files},
           'observations': [first], 'cache': str(cache), 'cache_policy': 'retained across process boots; first cache state recorded in log directory',
           'cache_files_before': sum(1 for path in cache.rglob('*') if path.is_file()),
           'scope': 'native kernel test environment; no serving image or throughput qualification'}
receipt['qualification_policy'] = {
    'hottest_zone_start_below_c': 55.0,
    'fan_service': 'dgx-fan-control.service',
    'fan_active_required_during_run': True,
    'host_memory_floor_gib': 48,
    'gpu_temperature_stop_above_c': 80,
    'transport': 'production cooling helpers with private local command adapter',
}
receipt['fan_commands'] = []
receipt['fan_curve'] = []
receipt['cooling'] = []
local_node = {'name': socket.gethostname()}


def local_command(nodes, node, *command):
    assert node == local_node, 'kernel cooling must only touch its local host'
    hold(beat=True)
    run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=45)
    receipt['fan_commands'].append({'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                    'command': list(command), 'returncode': run.returncode,
                                    'stdout': run.stdout, 'stderr': run.stderr})
    return run


cooling = helpers(local_command)
started = time.monotonic()
end = dt.datetime.fromisoformat(window['expected_end'].replace('Z', '+00:00')) if window['expected_end'] else None
process = None


def terminate(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
try:
    hold(beat=True)
    receipt['cooling'] = cooling.cool_nodes({}, [local_node], 55.0, 600.0)
    receipt['fan_curve'] = cooling.start_fan_curve({}, [local_node])
    problems = cooling.fan_curve_problems(receipt['fan_curve'])
    if problems:
        raise RuntimeError('; '.join(problems))
    ready = telemetry()
    receipt['observations'].append(ready)
    assert ready['performance_fan_active'], 'performance fan curve is not active'
    assert max(ready['thermal_zones_c'].values()) < 55, ready
    assert ready['host_available_bytes'] >= 48 * 2**30, ready
    receipt['fan_unit'] = local_command({}, local_node, 'systemctl', 'cat',
                                         'dgx-fan-control.service').stdout
    receipt['performance_state_before'] = local_command(
        {}, local_node, 'nvidia-smi', '-q', '-d', 'PERFORMANCE').stdout
    if end and dt.datetime.now(dt.timezone.utc) >= end - dt.timedelta(seconds=60):
        raise RuntimeError('window deadline reached during cooling')
    with (result / 'output.txt').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        last_beat = time.monotonic()
        while process.poll() is None:
            time.sleep(5)
            sample = telemetry();receipt['observations'].append(sample)
            if not sample['performance_fan_active']:
                raise RuntimeError('performance fan curve stopped during measurement')
            if sample['host_available_bytes'] < 48 * 2**30 or sample['temperature_c'] > 80:
                raise RuntimeError('memory/cooling guard reached')
            if time.monotonic() - last_beat >= 30:
                hold(beat=True);last_beat = time.monotonic()
            if time.monotonic() - started > 1500 or (end and dt.datetime.now(dt.timezone.utc) >= end - dt.timedelta(seconds=60)):
                raise RuntimeError('bounded GPU job/window deadline reached')
        receipt['exit_code'] = process.returncode
        final = telemetry()
        receipt['observations'].append(final)
        if not final['performance_fan_active']:
            raise RuntimeError('performance curve absent at measurement completion')
except BaseException as error:
    receipt['failure'] = repr(error)
    receipt.setdefault('exit_code', 1)
    raise
finally:
    try:
        if process is not None and process.poll() is None:
            subprocess.run(['docker', 'stop', '--time', '15', name], check=False, timeout=30)
            process.wait(timeout=30)
            receipt['exit_code'] = process.returncode or 1
    except BaseException as error:
        receipt['container_stop_error'] = repr(error)
        receipt['exit_code'] = 1
    try:
        receipt['performance_state_after'] = output('nvidia-smi', '-q', '-d', 'PERFORMANCE')
    except BaseException as error:
        receipt['performance_state_after_error'] = repr(error)
        receipt['exit_code'] = 1
    try:
        receipt['fan_curve_restore_failures'] = cooling.stop_fan_curve(
            {}, [local_node], receipt['fan_curve'])
        if receipt['fan_curve_restore_failures']:
            receipt['exit_code'] = 1
    except BaseException as error:
        receipt['fan_curve_restore_error'] = repr(error)
        receipt['exit_code'] = 1
    receipt['seconds'] = time.monotonic() - started
    (result / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('receipt', result, flush=True)
raise SystemExit(receipt['exit_code'])
