#!/usr/bin/env python3
"""Run the serving gate under the repository's identity, memory and thermal checks."""
import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from window_lease import require_window
from source_overlay import verify as verify_source

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f.config import configuration
from spark_ds41f.bench import (
    MemoryWatch, PeakWatch, bench_identity, cool_nodes, fan_control_enabled,
    fan_curve_problems, start_fan_curve, stop_fan_curve,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--cluster-config', required=True)
    parser.add_argument('--window-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, extra = parser.parse_known_args()
    require_window(args.window_state)
    cluster, nodes, _ = configuration(args)
    if not cluster['deployment']['launch_enabled']:
        raise SystemExit('Candidate launch is disabled; serving qualification is not enabled')
    identities = json.loads(Path(__file__).with_name('base.json').read_text())
    profiles = {'screen-tp3.json': 'candidate', 'control-tp3.json': 'control'}
    policy = profiles.get(Path(args.cluster_config).name)
    if policy is None:
        raise SystemExit('This screen requires the published candidate overlay or control profile')
    if '--recovery-policy' in extra:
        raise SystemExit('Recovery policy is selected from the verified source identity')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record_path = args.out.with_suffix('.conditions.json')
    with record_path.open('x') as record:
        identity = bench_identity(cluster, nodes, args.cluster_config, ROOT / args.cluster_config)
        conditions = {'identity': identity, 'fan_curve': [], 'halted': []}
        child = None
        def halt(reason):
            if reason not in conditions['halted']:
                conditions['halted'].append(reason)
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
        memory = None
        peak = None
        code = 1
        try:
            if identity['client_dirty'] or any(row['checkout'] != identity['client_commit'] for row in identity['nodes'].values()):
                raise RuntimeError('Client or serving-rank checkout differs from the clean published experiment')
            if identity['live_problems']:
                raise RuntimeError('; '.join(identity['live_problems']))
            conditions['effective_source'] = verify_source(
                cluster, nodes, policy, Path(__file__).with_name('overlay.json'))
            memory = MemoryWatch(nodes, float(cluster['host']['memguard_gib']) + 1, halt)
            conditions['cooling'] = cool_nodes(nodes, nodes['nodes'], 55, 900,
                                               fan_control=fan_control_enabled(cluster))
            conditions['fan_curve'] = start_fan_curve(nodes, nodes['nodes'],
                                                     fan_control=fan_control_enabled(cluster))
            problems = fan_curve_problems(conditions['fan_curve'])
            if problems:
                raise RuntimeError('; '.join(problems))
            memory.wait_ready()
            memory.check()
            if conditions['halted']:
                raise RuntimeError('; '.join(conditions['halted']))
            peak = PeakWatch(args.url)
            child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('probe.py')),
                                      args.url, '--out', str(args.out), '--window-state', str(args.window_state), '--recovery-policy', policy, *extra], cwd=ROOT)
            while child.poll() is None:
                try:
                    require_window(args.window_state)
                    memory.check()
                    if conditions['halted']:
                        raise RuntimeError('; '.join(conditions['halted']))
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
            code = child.returncode
        except (Exception, KeyboardInterrupt) as error:
            halt(repr(error))
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            conditions['fan_stop_failures'] = stop_fan_curve(nodes, nodes['nodes'], conditions['fan_curve'])
            if memory is not None:
                conditions['thermal'] = memory.thermal_summary()
                conditions['memory'] = memory.close()
            if peak is not None:
                conditions['peak'] = peak.finish()
            conditions['exit_code'] = code
            conditions['passed'] = code == 0 and not conditions['halted'] and not conditions['fan_stop_failures']
            json.dump(conditions, record, indent=2)
        return int(not conditions['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
