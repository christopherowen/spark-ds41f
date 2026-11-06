#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run on dgx1 after the preceding hold/request handoff, never beside its holder.

Uses the original clean deployment's normal publish/live/idle guards and
watchdog. The tested checkout is separate; no serving checkout is synchronized.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

ORIGINAL = Path('/home/swank/projects/spark-ds41f')
HOLDER = 'spark-lab: l2-access-retention-20261106'
PROFILE = 'experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json'
HOST_ROOT = '/home/swank/spark-lab/l2-access-retention'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', required=True)
    parser.add_argument('--boot', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.revision)
    checkout = Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() == args.revision
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain']).strip()
    sys.path.insert(0, str(ORIGINAL))
    from spark_ds41f import lab
    assert not lab.read_hold(), 'existing hold; wait for its holder'
    request = Path.home() / 'spark-request.json'
    pending = json.loads(request.read_text()) if request.exists() else None
    assert pending is None or pending['who'] == 'l2-access-retention-20261106', pending
    assert not subprocess.check_output(['git', '-C', str(ORIGINAL), 'status', '--porcelain']).strip()
    original_commit = subprocess.check_output(['git', '-C', str(ORIGINAL), 'rev-parse', 'HEAD'], text=True).strip()
    lab.PRODUCTION_CONFIG = PROFILE
    # Strengthen ownership checks in this controller: prefix sharing is not ownership.
    lab.hold_is_ours = lambda h: bool(h) and h.get('holder') == HOLDER
    atomic = lab.write_json_atomic
    claimed = False

    def guarded_write(path, data):
        nonlocal claimed
        if path == lab.HOLD:
            if not claimed:
                latest = json.loads(request.read_text()) if request.exists() else None
                assert latest is None or latest['who'] == 'l2-access-retention-20261106', latest
                with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
                    f.write(json.dumps(data, indent=2) + '\n')
                claimed = True
                return
            assert lab.hold_is_ours(lab.read_hold()), 'lost hold'
        atomic(path, data)

    lab.write_json_atomic = guarded_write
    lab.window_open(0, 'l2-access-retention-20261106')
    assert lab.hold_is_ours(lab.read_hold())
    if pending is not None and request.exists() and json.loads(request.read_text()) == pending:
        request.unlink()
    stop = threading.Event()

    def beat():
        while not stop.wait(20):
            lab.beat(f'GB10 retention screen boot {args.boot}; serving unchanged at {original_commit}')

    heartbeat = threading.Thread(target=beat, daemon=True)
    heartbeat.start()
    status = 1
    try:
        for boot in range(args.boot, 3):
            if boot != args.boot and request.exists():
                print('Yielding after the current job to the pending request.', flush=True)
                break
            script = f'{HOST_ROOT}/checkout/experiments/2026-11-06-l2-access-retention/node_job.sh'
            command = ['ssh', '-n', 'dgx4', 'bash', script, str(boot), args.revision, f"'{HOLDER}'"]
            print(json.dumps({'original_commit': original_commit, 'profile': PROFILE,
                              'revision': args.revision, 'command': command}), flush=True)
            status = subprocess.call(command)
            if status:
                break
    finally:
        stop.set()
        heartbeat.join(timeout=25)
        assert lab.hold_is_ours(lab.read_hold()), 'lost hold; cannot restore another window'
        assert subprocess.check_output(['git', '-C', str(ORIGINAL), 'rev-parse', 'HEAD'], text=True).strip() == original_commit
        if not lab.window_close('bounded retention screen finished'):
            raise RuntimeError('production verification failed; hold retained')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
