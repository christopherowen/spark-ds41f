#!/usr/bin/env python3
"""Bounded, local fabric lifetime around the generated collective probe.

Run on all four ranks only inside a coordinated window with serving stopped.
This runner deliberately cannot host serving: it owns one named probe container,
removes that container before retiring markers, and cleans only its own rules.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import mesh_fabric as fabric

PROBE = 'spark-collective-probe'


def execute(command):
    subprocess.run(command, check=True, timeout=20)


def wait_ready(process, device):
    if not select.select([process.stdout], [], [], 10)[0]:
        raise RuntimeError(f'{device}: marker readiness timed out')
    if process.stdout.readline().strip() != f'READY {device}' or process.poll() is not None:
        raise RuntimeError(f'{device}: marker failed to attach')


def stop_requested(sig, frame):
    raise InterruptedError(f'stopping mesh probe on signal {sig}')


def run(args):
    nodes = json.loads(args.nodes.read_text())
    plan = fabric.build_plan(nodes, args.paths)[args.rank]
    node = next(n for n in nodes['nodes'] if n['rank'] == args.rank)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if command[:3] != ['docker','run','--rm'] or f'--name={PROBE}' not in command:
        raise ValueError('only the generated named docker collective probe is supported')
    if os.geteuid() != 0:
        raise ValueError('the fabric probe requires sudo')
    # One host owner. Do not replace somebody else's route, filter or container.
    with open('/run/lock/spark-mesh-probe.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors, fixes = fabric.findings(node, fabric.inventory(node))
        if errors:
            raise ValueError('\n'.join(errors + fixes))
        for name in (args.serving_container, PROBE):
            if fabric.capture(['docker','ps','-aq','--filter',f'name=^{name}$']).strip():
                raise ValueError(f'remove the stopped {name} container before qualification; do not replace existing state')
        cleanup, markers, created_qdiscs = [], [], []
        child = None
        try:
            for rule in plan['rules']:
                dev = rule['ingress']
                qdiscs = json.loads(fabric.capture(['tc','-j','qdisc','show','dev',dev]))
                if not any(q.get('kind') in ('clsact','ingress') for q in qdiscs):
                    execute(['tc','qdisc','add','dev',dev,'clsact'])
                    created_qdiscs.append(dev)
                # Reserve an entire preference; tc add can otherwise append to it.
                existing = json.loads(fabric.capture(['tc','-j','filter','show','dev',dev,'ingress']))
                if any(f.get('pref') == rule['pref'] for f in existing):
                    raise ValueError(f"{dev}: TC preference {rule['pref']} is already owned")
                execute(rule['add']); cleanup.append(rule['delete'])
                installed = json.loads(fabric.capture(['tc','-j','filter','show','dev',dev,'ingress']))
                if not fabric.offload_ready(rule, installed):
                    raise RuntimeError(f'{dev}: forwarding rule was not offloaded to hardware')
            for route in plan['routes']:
                execute(route['add']); cleanup.append(route['delete'])
            for spec in plan['markers']:
                marker = subprocess.Popen([str(args.marker), spec['device'], spec['source_ip'], spec['destination_ip']], stdout=subprocess.PIPE, text=True)
                markers.append(marker); wait_ready(marker, spec['device'])
            print(json.dumps({'rank': args.rank, 'hardware_rules_ready': True,
                              'markers_ready': plan['markers']}), flush=True)
            child = subprocess.Popen(command)
            deadline = time.monotonic() + 660
            while child.poll() is None:
                if any(m.poll() is not None for m in markers):
                    raise RuntimeError('RDMA marker died; stopping the dependent probe')
                if time.monotonic() >= deadline:
                    raise TimeoutError('bounded collective probe exceeded 660 seconds')
                time.sleep(.2)
            return child.returncode
        finally:
            # Always stop dependent QPs before tearing down the NIC paths.
            # Ignore further TERM/INT while completing this cleanup.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if child is not None:
                # The preflight and host lock establish ownership of this name.
                while True:
                    try:
                        result = subprocess.run(['docker','rm','-f',PROBE], timeout=30, capture_output=True, text=True)
                        if result.returncode == 0 or 'No such container' in result.stderr:
                            break
                        error = result.stderr
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        error = str(exc)
                    print('Cannot stop owned probe; retaining paths and retrying: '+error, flush=True)
                    time.sleep(5)
                # The container is gone. A stuck Docker client no longer owns QPs.
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
            for marker in markers:
                if marker.poll() is None: marker.terminate()
            for marker in markers:
                try: marker.wait(timeout=5)
                except subprocess.TimeoutExpired: marker.kill(); marker.wait(timeout=5)
            failures = []
            # Read offloaded packet counters before retiring the owned rules.
            # A telemetry failure must not prevent path cleanup.
            if child is not None:
                for rule in plan['rules']:
                    try:
                        counters = json.loads(fabric.capture([
                            'tc', '-j', '-s', 'filter', 'show', 'dev',
                            rule['ingress'], 'ingress']))
                        print(json.dumps({'rank': args.rank, 'forwarding_counters':
                            [f for f in counters if f.get('pref') == rule['pref']],
                            'device': rule['ingress']}), flush=True)
                    except (subprocess.SubprocessError, OSError, ValueError) as exc:
                        failures.append('counter read failed: ' + str(exc))
            for undo in reversed(cleanup):
                try: execute(undo)
                except (subprocess.SubprocessError, OSError) as e: failures.append(str(e))
            # Keep a shared clsact if another owner attached a rule meanwhile.
            for dev in created_qdiscs:
                try:
                    ingress = json.loads(fabric.capture(['tc','-j','filter','show','dev',dev,'ingress']))
                    egress = json.loads(fabric.capture(['tc','-j','filter','show','dev',dev,'egress']))
                    if not ingress and not egress: execute(['tc','qdisc','del','dev',dev,'clsact'])
                except (subprocess.SubprocessError, OSError, ValueError) as e: failures.append(str(e))
            if failures: raise RuntimeError('fabric cleanup incomplete: '+'; '.join(failures))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes', type=Path, required=True)
    p.add_argument('--paths', type=int, choices=(2, 4), default=2)
    p.add_argument('--rank', type=int, choices=range(4), required=True)
    p.add_argument('--marker', type=Path, required=True)
    p.add_argument('--serving-container', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
