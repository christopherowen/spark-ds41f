#!/usr/bin/env python3
"""Restore a captured deployment commit/profile for one explicitly owned window.

State is captured and reviewed on the head before a branch checkout. This helper
does not acquire a window, authorize a run, change topology, or refresh heartbeats.
"""
import argparse
import datetime as dt
import fcntl
import json
import subprocess
import time
from pathlib import Path


def read_hold(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def ours(state, hold):
    return bool(hold) and hold.get('window') == state['window'] and hold.get('holder') == state['holder']


def stale(hold, now=None):
    now = time.time() if now is None else now
    stamp = dt.datetime.fromisoformat(hold['heartbeat'].replace('Z', '+00:00')).timestamp()
    return now - stamp > 900


def checked(command, cwd, output=None):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=3600)
    if output is not None:
        output.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{command}: exit {result.returncode}; {result.stdout[-2000:]}')
    return result.stdout


def verify_plan(action, text):
    if '--apply would refuse' in text or 'ERROR:' in text:
        raise RuntimeError(f'{action}: refusing plan')
    if action in ('stop', 'start') and not any(line.startswith('live:') for line in text.splitlines()):
        raise RuntimeError(f'{action}: plan has no live-state line')
    if action == 'start':
        shape = [line for line in text.splitlines() if line.startswith('shape:')]
        if len(shape) != 1 or 'CHANGES' in shape[0]:
            raise RuntimeError('start: missing or changed shape')


def restore(state, hold_path, output):
    controller = Path(state['restore_controller'])
    output.mkdir(parents=True, exist_ok=False)
    def own():
        if not ours(state, read_hold(hold_path)):
            raise RuntimeError('Window ownership changed; no further action')
    own()
    if checked(['git', 'rev-parse', 'HEAD'], controller).strip() != state['restore_commit']:
        raise RuntimeError('Restore controller moved from the captured commit')
    if checked(['git', 'status', '--porcelain', '--untracked-files=normal'], controller).strip():
        raise RuntimeError('Restore controller is dirty')
    actual = checked(['docker', 'image', 'inspect', state['restore_image_tag'], '--format', '{{.Id}}'],
                     controller).strip()
    if actual != state['restore_image_id']:
        raise RuntimeError('Restore image tag no longer names the captured digest')
    base = [str(controller / 'bin/spark'), '--cluster-config', state['restore_profile']]
    checked([*base, 'doctor'], controller, output / 'doctor-offline.txt')
    steps = [('stop', []), ('sync', ['--revision', state['restore_commit']]), ('start', ['--replace'])]
    for action, extra in steps:
        own()
        argv = [*base, 'cluster', action, *extra]
        plan = checked(argv, controller, output / f'{action}-plan.txt')
        verify_plan(action, plan)
        own()
        checked([*argv, '--apply'], controller, output / f'{action}-apply.txt')
    own()
    checked([*base, 'doctor', '--live'], controller, output / 'doctor-live.txt')
    own()
    (output / 'restored.json').write_text(json.dumps({
        'window': state['window'], 'commit': state['restore_commit'],
        'profile': state['restore_profile'], 'image': actual, 'restored': True}, indent=2) + '\n')
    # A crashed coordinator may leave a build/probe child alive. A marker is
    # removed only after that job is confirmed finished; never release to a
    # different GPU holder while an owned job is unresolved.
    active = [p for p in state.get('activity_markers', []) if Path(p).exists()]
    if active:
        raise RuntimeError(f'Service restored, but unresolved job markers keep the hold: {active}')
    # Both the original commit and profile have been restored before release.
    hold_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=Path)
    parser.add_argument('--restore-now', action='store_true')
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    hold_path = Path(state['hold_path'])
    if not args.restore_now and ours(state, read_hold(hold_path)):
        import os
        args.state.with_suffix('.watchdog-ready.json').write_text(json.dumps({
            'window': state['window'], 'pid': os.getpid(), 'ready': True}) + '\n')
        print(f'Watchdog ready for {state["window"]}', flush=True)
    while True:
        hold = read_hold(hold_path)
        if not ours(state, hold):
            return 0
        if args.restore_now or stale(hold):
            with Path(state['operation_lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                current = read_hold(hold_path)
                if not ours(state, current):
                    return 0
                if not args.restore_now and not stale(current):
                    continue
                # The operator holds this same lock around each coordinated
                # operation, so restoration cannot race an in-flight start/sync.
                output = Path(state['restore_receipts']) / str(time.time_ns())
                try:
                    restore(state, hold_path, output)
                    return 0
                except Exception as error:
                    output.mkdir(parents=True, exist_ok=True)
                    (output / 'failure.txt').write_text(repr(error) + '\n')
                    print(f'Restore failed; hold retained: {error}', flush=True)
                    return 1
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
