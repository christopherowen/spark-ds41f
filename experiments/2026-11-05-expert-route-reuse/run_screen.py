"""Remote-only wrapper retaining kernel-local locks, refusal and receipts.

The coordinator must read the dgx1 hold immediately before invoking this script.
This script requires an already acquired, exactly named node-local hold. It does
not acquire a window, change serving state, or install/configure fan control.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import bench, lab


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=20).stdout.strip()


def thermal():
    readings = [int(p.read_text()) / 1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')]
    if not readings:
        raise RuntimeError('no thermal-zone readings')
    return max(readings)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('no MemAvailable reading')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('gate', 'contract-before', 'default', 'smoke', 'full'), required=True)
    parser.add_argument('--process-id', required=True)
    parser.add_argument('--holder', required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.process_id):
        raise ValueError('process ID must be a simple container-name component')
    bundle = Path(__file__).parent / 'bundles/swizzle-screen'
    name = 'expert-swizzle-' + args.process_id
    # Retain the shared GPU lock but refuse contention instead of waiting
    # behind another job after the initial ownership/resource checks.
    original_flock = lab.fcntl.flock
    lab.fcntl.flock = lambda fd, flags: original_flock(fd, flags | lab.fcntl.LOCK_NB)
    original = lab.bundle_command
    def bounded(bundle, candidate, out):
        result = original(bundle, candidate, out)
        if args.mode == 'contract-before':
            destination = ':/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro'
            hits = [i for i, value in enumerate(result) if value.endswith(destination)]
            assert len(hits) == 1
            result[hits[0]] = str(bundle / 'control_moe.py') + destination
        result[2:2] = ['-v', str(out) + ':/receipt', '--name', name, '--label', 'expert-route-window=' + args.window, '--network', 'none', '--cpus', '2',
                       '--memory', '16g', '--memory-swap', '16g']
        return result + ['--mode', args.mode, '--process-id', args.process_id]
    lab.bundle_command = bounded
    if args.dry_run:
        return lab.kernel_local(str(bundle), args.out, True)
    if command('git', '-C', str(ROOT), 'status', '--porcelain'):
        raise RuntimeError('deployment checkout must be clean')
    revision = command('git', '-C', str(ROOT), 'rev-parse', 'HEAD')
    if not command('git', '-C', str(ROOT), 'branch', '-r', '--contains', revision):
        raise RuntimeError('deployment commit has no published remote reference')
    hold_path = Path.home() / 'spark-hold.json'
    def own_hold():
        hold = json.loads(hold_path.read_text())
        if hold.get('holder') != args.holder or hold.get('window') != args.window:
            raise RuntimeError('exclusive hold identity differs')
        return hold
    own_hold()
    if lab.container_running():
        raise RuntimeError('serving container present')
    fan = subprocess.run(['systemctl', 'is-active', 'dgx-fan-control.service'],
                         capture_output=True, text=True, timeout=15)
    performance = args.mode not in ('gate', 'contract-before')
    if not performance and thermal() >= 55:
        raise RuntimeError('cool below 55 C before starting this correctness process')
    if command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader').strip():
        raise RuntimeError('another GPU compute process is present')
    if available() < 24 * 1024**3:
        raise RuntimeError('less than 24 GiB host memory available')
    # Use the repository's existing fan/cooling implementation on this node.
    # Only its SSH transport is replaced with a checked local subprocess.
    node = {'name': socket.gethostname()}
    nodes = {'nodes': [node]}
    def local_transport(_nodes, target, *argv):
        if target['name'] != node['name']:
            raise RuntimeError('fan transport must stay on this node')
        return subprocess.run(argv, capture_output=True, text=True, timeout=30)
    bench.run_ssh = local_transport
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'deployment_commit': revision, 'process_id': args.process_id,
             'mode': args.mode, 'holder': args.holder, 'window': args.window,
             'cpu_limit': 2, 'memory_limit_gib': 16,
             'fan_service': fan.stdout.strip(), 'fan_service_before': fan.stdout.strip(),
             'performance_mode': performance, 'start_c': thermal(),
             'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'started': dt.datetime.now(dt.timezone.utc).isoformat(),
             'stop_reason': None, 'samples': [], 'fan_curve': [], 'cooling': []}
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                hold = own_hold()
                now = dt.datetime.now(dt.timezone.utc).isoformat()
                hold['heartbeat'] = now
                temporary = hold_path.with_suffix('.expert-tmp')
                temporary.write_text(json.dumps(hold, indent=2) + '\n')
                os.replace(temporary, hold_path)
                temp, free = thermal(), available()
                state['samples'].append({'at': now, 'hottest_zone_c': temp,
                                         'mem_available_bytes': free})
                if temp >= 85 or free < 12 * 1024**3:
                    raise RuntimeError(f'resource guard: {temp} C, {free} available bytes')
                if time.monotonic() - began > 2700:
                    raise RuntimeError('45-minute process limit')
            except Exception as exc:
                state['stop_reason'] = repr(exc)
                subprocess.run(['docker', 'stop', '--time', '10', name],
                               capture_output=True, timeout=20)
                return
            finally:
                (out / 'resources.json').write_text(json.dumps(state, indent=2) + '\n')
            stop.wait(5)
    began = time.monotonic()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        if performance:
            state['cooling'] = bench.cool_nodes(nodes, [node], 55, 600)
            state['fan_curve'] = bench.start_fan_curve(nodes, [node])
            problems = bench.fan_curve_problems(state['fan_curve'])
            if problems:
                raise RuntimeError('; '.join(problems))
            state['fan_service'] = 'active'
        own_hold()
        if state['stop_reason']:
            raise RuntimeError(state['stop_reason'])
        result = lab.kernel_local(str(bundle), str(out), False)
        if state['stop_reason']:
            raise RuntimeError(state['stop_reason'])
        if args.mode == 'gate':
            lines = (out / 'output.txt').read_text().splitlines()
            gate = lab.suite_verdict(lines)
            gate['passed'] = bool(gate['passed'] and re.search(r'\b24 passed\b', gate['summary'])
                                  and not re.search(r'\b\d+ skipped\b', gate['summary']))
            receipt = json.loads((out / 'verdict.json').read_text())
            receipt.update(gate)
            receipt['passed'] = gate['passed'] and receipt['exit'] == 0
            (out / 'verdict.json').write_text(json.dumps(receipt, indent=2) + '\n')
            if not gate['passed']:
                raise RuntimeError('native gate did not pass tests: ' + str(gate))
        return result
    finally:
        state['fan_stop_failures'] = bench.stop_fan_curve(nodes, [node], state['fan_curve'])
        stop.set(); watcher.join(timeout=25)
        state['finished'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (out / 'resources.json').write_text(json.dumps(state, indent=2) + '\n')
        if state['fan_stop_failures']:
            raise RuntimeError('fan restoration failed: ' + str(state['fan_stop_failures']))


if __name__ == '__main__':
    raise SystemExit(main())
