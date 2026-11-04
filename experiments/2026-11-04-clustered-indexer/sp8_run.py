"""Run published isolated kernel bundles on sp8 with an exclusive host lease."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab

HERE = Path(__file__).resolve().parent
IMAGE = json.loads((HERE / 'sp8.json').read_text())['image_id']
HOLD = Path.home() / 'spark-hold.json'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def temperature():
    samples = []
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            value = float(path.read_text()) / 1000
            if 0 < value < 200:
                samples.append(value)
        except (OSError, ValueError):
            pass
    return max(samples) if samples else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jobs', nargs='+', help='bundle:unique-run-name, in order')
    options = parser.parse_args()
    if socket.gethostname() != 'spark-0016':
        raise SystemExit('this continuation is scoped to sp8 only')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise SystemExit('published checkout must be clean')
    if subprocess.check_output(['docker', 'ps', '-q'], text=True).strip():
        raise SystemExit('sp8 must have no running container before this isolated batch')
    if memory() < 32 * 2**30 or (temperature() or 0) >= 55:
        raise SystemExit('sp8 must start cool with at least 32 GiB available')
    token = uuid.uuid4().hex
    hold = {'holder': 'clustered-indexer-sp8', 'token': token, 'since': stamp(),
            'heartbeat': stamp(), 'expected_end': None, 'scope': ['sp8'],
            'rule': 'exclusive isolated GPU tests on sp8; no serving changes',
            'commit': commit}
    # O_EXCL refuses an existing foreign reservation instead of overwriting it.
    with HOLD.open('x') as handle:
        json.dump(hold, handle, indent=2)
    stop = threading.Event()
    low_memory = threading.Event()
    observations = {'minimum_mem_available_bytes': memory(),
                    'maximum_thermal_c': temperature()}
    lease_error = []

    def monitor():
        while not stop.wait(1):
            available, thermal = memory(), temperature()
            observations['minimum_mem_available_bytes'] = min(
                observations['minimum_mem_available_bytes'], available)
            if thermal is not None:
                observations['maximum_thermal_c'] = max(
                    observations['maximum_thermal_c'] or thermal, thermal)
            if available < 24 * 2**30:
                low_memory.set()
            try:
                current = json.loads(HOLD.read_text())
                if current.get('token') != token:
                    raise RuntimeError('sp8 reservation ownership changed')
                current['heartbeat'] = stamp()
                temp = HOLD.with_name('.spark-hold-' + token + '.tmp')
                temp.write_text(json.dumps(current, indent=2) + '\n')
                os.replace(temp, HOLD)
            except Exception as error:
                lease_error.append(str(error))
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    success = True
    container = None
    try:
        with open('/tmp/spark-lab-gpu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for job in options.jobs:
                if (Path.home() / 'spark-request.json').exists():
                    print('pending request: returning sp8 after the current completed job', flush=True)
                    break
                bundle_name, run = job.split(':', 1)
                if not bundle_name.startswith('sp8-') or Path(run).name != run:
                    raise ValueError('expected a local sp8 bundle and a single-component run name')
                bundle = HERE / bundle_name
                candidate = json.loads((bundle / 'candidate.json').read_text())
                assert candidate['image'] == IMAGE, 'sp8 image identity differs'
                out = Path.home() / 'spark-lab/results/clustered-indexer' / run
                out.mkdir(parents=True, exist_ok=False)
                container = 'clustered-indexer-' + token[:12]
                lab.compile_cache(candidate).mkdir(parents=True, exist_ok=True)
                began = stamp()
                command = lab.bundle_command(bundle, candidate, out)
                # A bounded, uniquely named kernel test; never a serving launch.
                command[2:2] = ['--name', container, '--memory', '16g',
                                '--memory-swap', '16g', '--network', 'none']
                mounted = {}
                for source, _ in lab.bundle_mounts(bundle, candidate):
                    if source.is_file():
                        mounted[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                started = time.monotonic()
                (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
                with (out / 'output.txt').open('w') as handle:
                    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        if low_memory.is_set() or lease_error or time.monotonic() - started > 900:
                            subprocess.run(['docker', 'stop', '-t', '5', container],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            process.wait(timeout=15)
                            break
                        time.sleep(0.2)
                output = (out / 'output.txt').read_text()
                verdict = lab.verdict_from_lines(output.splitlines(), candidate)
                verdict.update(node=socket.gethostname(), image=IMAGE, commit=commit,
                               command=command, exit=process.returncode,
                               started_at=began, ended_at=stamp(),
                               seconds=round(time.monotonic() - started, 3),
                               mounted_inputs_sha256=mounted, memory=dict(observations),
                               lease_errors=list(lease_error))
                verdict['passed'] &= process.returncode == 0 and not low_memory.is_set() and not lease_error
                (out / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
                for line in reversed(output.splitlines()):
                    if line.startswith('RESULT '):
                        (out / 'result.json').write_text(json.dumps(json.loads(line[7:]), indent=2) + '\n')
                        break
                print(json.dumps({'run': run, 'passed': verdict['passed'],
                                  'seconds': verdict['seconds'], 'out': str(out)}), flush=True)
                container = None
                if not verdict['passed']:
                    success = False
                    break
    finally:
        if container:
            subprocess.run(['docker', 'stop', '-t', '5', container],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stop.set()
        watcher.join(timeout=5)
        if HOLD.exists() and json.loads(HOLD.read_text()).get('token') == token:
            HOLD.unlink()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
