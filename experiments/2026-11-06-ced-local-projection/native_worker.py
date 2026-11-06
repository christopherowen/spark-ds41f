"""One token-owned Spaceport native correctness process; no timing mode exists."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

PARENT = 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'
HOLDER = 'ced-sp567-native-correctness'
NODES = ['sp5', 'sp6', 'sp7']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--token', required=True)
parser.add_argument('--boot', required=True)
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
source = args.source.resolve()
identity = json.loads((EXP/'identity.json').read_text())


def output(*command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True, timeout=30).strip()


def lease():
    hold = json.loads((Path.home()/'spark-hold.json').read_text())
    assert hold.get('token') == args.token and hold.get('holder') == HOLDER, 'foreign lease'
    assert hold.get('nodes') == NODES and hold.get('phase') == 'native', 'cohort is not running native tests'
    return hold


def telemetry():
    zones = {p.parent.name: int(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    assert zones, 'thermal zones unreadable'
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    raw = output('nvidia-smi', '--query-gpu=uuid,temperature.gpu,clocks.gr', '--format=csv,noheader,nounits')
    uuid, temperature, clock = raw.split(',')
    return {'at': time.time(), 'thermal_zones_c': zones, 'host_available_bytes': available,
            'gpu_uuid': uuid.strip(), 'gpu_temperature_c': float(temperature), 'graphics_mhz': clock.strip()}


held = lease()
assert not output('git', 'status', '--porcelain', cwd=ROOT), 'dirty harness checkout'
assert not output('git', 'status', '--porcelain', cwd=source), 'dirty source checkout'
assert output('git', 'rev-parse', 'HEAD', cwd=ROOT) == held['controller_commit']
assert output('git', 'rev-parse', 'HEAD', cwd=source) == identity['candidate']['patch_head']
assert output('git', 'rev-parse', 'HEAD^{tree}', cwd=source) == identity['candidate']['expected_tree']
labels = json.loads(output('docker', 'image', 'inspect', PARENT))[0]['Config']['Labels']
assert labels['local.spark-ds41f.vllm.tree'] == identity['unreleased_control']['tree']
manifest = json.loads((EXP/'source.json').read_text())
for project in ('b12x', 'nccl', 'tilelang', 'tile_kernels', 'sparknet'):
    assert labels[f'local.spark-ds41f.{project}.tree'] == manifest[project]['expected_tree'], project
assert not output('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'), 'GPU is busy'
changed = output('git', 'diff', '--name-only', identity['unreleased_control']['head'], 'HEAD', cwd=source).splitlines()
files = [name for name in changed if name.startswith('vllm/')]
assert len(files) == 6 and all(name.endswith('.py') for name in files)
base = Path.home()/'spark-lab/ced-sp567/results'/args.token
result = base/args.boot
result.mkdir(parents=True, exist_ok=False)
cache = Path.home()/'spark-lab/ced-sp567/cache'
cache.mkdir(parents=True, exist_ok=True)
name = 'ced-sp567-'+args.token[:12]+'-'+args.boot
command = ['docker', 'run', '--rm', '--name', name, '--gpus=all', '--cpus=1', '--memory=16g',
           '--memory-swap=16g', '--pids-limit=256', '--network=none', '--shm-size=1g',
           '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MAX_JOBS=1',
           '-e', 'NVCC_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'VLLM_CACHE_ROOT=/root/.cache/vllm',
           '-v', str(source)+':/src:ro', '-v', str(EXP)+':/check:ro', '-v', str(cache)+':/root/.cache:rw']
for name_in_source in files:
    command += ['-v', str(source/name_in_source)+':/opt/spark-ds41f/candidate/vllm/'+name_in_source+':ro']
command += ['--entrypoint', 'python3', PARENT, '/check/gpu.py']
record = {'host': socket.gethostname(), 'lease': held, 'source': identity['candidate'],
          'parent_image': PARENT, 'parent_labels': labels, 'command': command,
          'source_file_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
          'scope': 'native correctness only; no serving, distributed-collective or performance claim',
          'policy': {'start_hottest_below_c': 55, 'stop_hottest_above_c': 70,
                     'stop_gpu_above_c': 70, 'host_memory_floor_gib': 48, 'process_limit_seconds': 600,
                     'cooling_timeout_seconds': 600, 'fan': 'unchanged stock cooling; no timing qualification'},
          'observations': [], 'exit_code': 1}
process = None
began = time.monotonic()
try:
    # Passive cooling is acceptable only for bit checks; record every wait.
    while True:
        lease(); sample = telemetry(); record['observations'].append(sample)
        assert sample['host_available_bytes'] >= 48*2**30, sample
        if max(sample['thermal_zones_c'].values()) < 55:
            break
        if time.monotonic()-began > 600:
            raise RuntimeError('native cooling baseline not reached')
        time.sleep(5)
    lease()
    with (result/'output.txt').open('w') as log:
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(5)
            lease(); sample = telemetry(); record['observations'].append(sample)
            if max(sample['thermal_zones_c'].values()) > 70 or sample['gpu_temperature_c'] > 70:
                raise RuntimeError('native thermal guard reached')
            if sample['host_available_bytes'] < 48*2**30:
                raise RuntimeError('native host-memory guard reached')
            if time.monotonic()-started > 600:
                raise RuntimeError('native process deadline reached')
        record['exit_code'] = process.returncode
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    try:
        if process is not None and process.poll() is None:
            stopped = subprocess.run(['docker', 'stop', '--time=10', name], capture_output=True, text=True, timeout=25)
            record['stop'] = {'returncode': stopped.returncode, 'stdout': stopped.stdout, 'stderr': stopped.stderr}
            process.wait(timeout=30)
    except BaseException as error:
        record['cleanup_error'] = repr(error)
        record['exit_code'] = 1
    record['elapsed_seconds'] = time.monotonic()-began
    (result/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print('receipt', result, flush=True)
raise SystemExit(record['exit_code'])
