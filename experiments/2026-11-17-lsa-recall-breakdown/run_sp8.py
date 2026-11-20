"""Single-GPU sp8 runner for the LSA recall breakdown: lease, one container, receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

HOLD_PATH = Path.home() / 'spark-hold.json'
JOB = 'experiments/2026-11-06-optional-lsa/sp8-bench-boot1/candidate.json'
DROPPED = '/tmp/benchmark_matrix.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def sh(*argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def gpu():
    return sh('nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.graphics,power.draw',
              '--format=csv,noheader')


def memory():
    apps = sh('nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader')
    return {'at': now(), 'gpu_apps': apps.splitlines() if apps else [], 'free_b': sh('free', '-b')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--source-commit', required=True,
                        help='commit of the LSA sources to mount, exported from the checkout')
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checkout, script = args.checkout.resolve(), args.script.resolve()
    args.out.mkdir(parents=True)
    commit = sh('git', 'rev-parse', args.source_commit + '^{commit}', cwd=checkout)
    # The mounted sources are exported at one recorded commit, so the checkout's
    # current branch and working tree are never an input.
    tree = script.parent / f'sources-{commit[:12]}'
    if not tree.exists():
        tree.mkdir()
        archive = subprocess.run(['git', 'archive', commit, 'experiments'], cwd=checkout,
                                 check=True, capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', str(tree)], input=archive, check=True)
    spec = json.loads((tree / JOB).read_text())

    def resolve(src):
        if src.startswith('~/'):
            return str(Path.home() / src[2:])
        return src if src.startswith('/') else str(tree / src)

    token = f'lsa-recall-breakdown-{uuid.uuid4().hex[:12]}'
    name = f'lsa-recall-{token[-12:]}'
    cache = Path.home() / '.cache/spark-lab/compile'
    argv = ['docker', 'run', '--name', name, '--network', 'none', '--cpus', '2', '--memory', '16g',
            '--pids-limit', '256', '--rm', '--gpus', 'all', '--ipc=host', '--ulimit=core=0']
    for key, value in sorted(spec['env'].items()):
        argv += ['-e', f'{key}={value}']
    for src, dst in spec['mounts']:
        if dst != DROPPED:
            argv += ['-v', f'{resolve(src)}:{dst}:ro']
    argv += ['-v', f'{script}:/tmp/breakdown.py:ro', '-v', f'{cache}:/c',
             '-w', '/opt/spark-ds41f/candidate/vllm', '--entrypoint', 'python3', spec['image'],
             '/tmp/breakdown.py', '--inputs', '/inputs', '--replay-receipt', '/replay.json']

    since = now()
    lease = {
        'holder': 'lsa-recall-breakdown', 'token': token, 'nodes': ['sp8'], 'pool': 'sp8-single',
        'since': since, 'expected_end': None, 'heartbeat': since,
        'controller': str(script.parent), 'controller_commit': commit,
        'restore_profile': 'none (standalone sp8 selection proxy; no serving container on sp8)',
        'rule': 'Exclusive sp8 single-GPU window for the untimed LSA recall breakdown; '
                'stale if heartbeat > 15m old.',
    }
    fd = os.open(str(HOLD_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(lease, indent=2) + '\n')
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30.0):
            try:
                current = json.loads(HOLD_PATH.read_text())
                if current.get('token') == token:
                    current['heartbeat'] = now()
                    tmp = HOLD_PATH.with_suffix('.tmp')
                    tmp.write_text(json.dumps(current, indent=2) + '\n')
                    tmp.replace(HOLD_PATH)
            except Exception:
                pass

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        (args.out / 'lease.json').write_text(json.dumps(lease, indent=2) + '\n')
        (args.out / 'command.json').write_text(json.dumps({
            'argv': argv, 'at': now(), 'lsa_source_checkout': str(checkout), 'lsa_source_commit': commit, 'lsa_source_tree': str(tree),
            'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
        (args.out / 'memory-before.json').write_text(json.dumps(memory()) + '\n')
        start_gpu, start = gpu(), time.monotonic()
        proc = subprocess.run(argv, capture_output=True, text=True)
        elapsed = time.monotonic() - start
        (args.out / 'output.txt').write_text(proc.stdout + ('\n' + proc.stderr if proc.stderr else ''))
        (args.out / 'verdict.json').write_text(json.dumps({
            'exit': proc.returncode, 'seconds': elapsed, 'at': now(),
            'start_gpu': start_gpu, 'end_gpu': gpu()}, indent=2) + '\n')
        (args.out / 'memory-after.json').write_text(json.dumps(memory()) + '\n')
        results = [line[7:] for line in proc.stdout.splitlines() if line.startswith('RESULT ')]
        if results:
            (args.out / 'breakdown.json').write_text(json.dumps(json.loads(results[-1]), indent=2) + '\n')
    finally:
        stop.set()
        try:
            if json.loads(HOLD_PATH.read_text()).get('token') == token:
                HOLD_PATH.unlink()
        except Exception:
            pass
        remaining = sh('docker', 'ps', '-a', '--filter', f'name={name}', '--format', '{{.Names}}')
        (args.out / 'released.json').write_text(json.dumps({
            'at': now(), 'hold_removed': not HOLD_PATH.exists(),
            'remaining_containers': remaining.splitlines() if remaining else []}, indent=2) + '\n')
    print(f'EXIT {proc.returncode} in {elapsed:.1f}s', flush=True)
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
