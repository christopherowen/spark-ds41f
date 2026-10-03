#!/usr/bin/env python3
"""Apply one reviewed data-NIC queue change on an idle host in an owned window."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import mesh_fabric as fabric


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    print(json.dumps({'command': command, 'exit': result.returncode,
                      'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
    result.check_returncode()
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=range(4), required=True)
    parser.add_argument('--expected', type=int, choices=(1024, 8192), required=True)
    parser.add_argument('--size', type=int, choices=(1024, 8192), required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or args.expected == args.size:
        raise ValueError('requires sudo and a different before/after size')
    nodes = json.loads(args.nodes.read_text())
    node = next(n for n in nodes['nodes'] if n['rank'] == args.rank)
    if subprocess.check_output(['hostname', '-s'], text=True).strip() != node['name']:
        raise ValueError('rank does not match host')
    # The coordinator validates the head-node hold and all hosts before dispatch.
    # Take the same host lock as the bounded forwarding runner.
    with open('/run/lock/spark3-mesh-probe.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if run(['docker', 'ps', '-q']).strip():
            raise ValueError('containers are running')
        if run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
            raise ValueError('GPU clients are running')
        if json.loads(run(['rdma', '-j', 'resource', 'show', 'ctx'])):
            raise ValueError('RDMA user contexts are open')
        node['mesh_hairpin_queue_size'] = args.expected
        before = fabric.inventory(node)
        errors, _ = fabric.findings(node, before)
        if errors:
            raise ValueError('\n'.join(errors))
        for port in node['mesh_ports'].values():
            for direction in ('ingress', 'egress'):
                if json.loads(run(['tc', '-j', 'filter', 'show', 'dev', port['netdev'], direction])):
                    raise ValueError('TC filters are present')
        print(json.dumps({'before_inventory': before}), flush=True)
        for port in node['mesh_ports'].values():
            device = 'pci/' + port['pci']
            old = json.loads(run(['devlink', '-s', '-j', 'dev', 'show', device]))
            count = old['dev'][device]['stats']['reload']['driver_reinit']['unspecified']
            run(['devlink', 'dev', 'param', 'set', device, 'name', 'hairpin_queue_size',
                 'value', str(args.size), 'cmode', 'driverinit'])
            run(['devlink', 'dev', 'reload', device, 'action', 'driver_reinit'])
            new = json.loads(run(['devlink', '-s', '-j', 'dev', 'show', device]))
            state = new['dev'][device]
            if state.get('reload_failed') or state['stats']['reload']['driver_reinit']['unspecified'] <= count:
                raise RuntimeError('driver reinitialization was not confirmed')
        print(json.dumps({'rank': args.rank, 'applied_queue_size': args.size,
                          'next': 'coordinator must recheck addresses, GIDs and NIC inventory on all hosts'}), flush=True)


if __name__ == '__main__':
    main()
