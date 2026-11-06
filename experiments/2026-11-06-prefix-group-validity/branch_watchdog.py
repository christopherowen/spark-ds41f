#!/usr/bin/env python3
"""Restore a captured deployment commit/profile for one explicitly owned window.

State is captured and reviewed on the head before a branch checkout. This helper
does not acquire a window, authorize a run, change topology, or refresh heartbeats.
"""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class ManagementLossError(RuntimeError):
    """Raised when SSH, daemon, or guard telemetry is lost."""


MANAGEMENT_LOSS_TOKENS = (
    'management_loss',
    'management loss',
    'telemetry loss',
    'telemetry',
    'guard loss',
    'guard_loss',
    'timed out',
    'timeout',
    'unreachable',
    'connection refused',
    'connection reset',
    'broken pipe',
    'no route to host',
    'operation timed out',
    'ssh:',
    'exit 255',
    'returncode 255',
    'unable to query git state',
    'cannot connect to the docker daemon',
    'docker daemon',
    'memguard',
    'fan_stop_failures',
)


def read_hold(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def ours(state, hold, review=False):
    if not isinstance(hold, dict) or hold.get('window') != state['window']:
        return False
    token = state.get('owner_token')
    if (token is not None or 'owner_token' in hold) and hold.get('owner_token') != token:
        return False
    if review and hold.get('original_holder') == state['holder']:
        return True
    return hold.get('holder') == state['holder']


def is_management_loss(error):
    if isinstance(error, (ManagementLossError, subprocess.TimeoutExpired)):
        return True
    text = str(error).lower()
    return any(token in text for token in MANAGEMENT_LOSS_TOKENS)


@contextmanager
def hold_mutation_lock(hold_path):
    lock_path = hold_path.with_name(hold_path.name + '.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write_hold_atomic(hold_path, payload):
    with tempfile.NamedTemporaryFile(dir=hold_path.parent, prefix='.hold-', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write((json.dumps(payload, indent=2) + '\n').encode())
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, hold_path)


def latch_management_loss(state, hold_path, reason, phase='watchdog', details=None):
    with hold_mutation_lock(hold_path):
        hold = read_hold(hold_path)
        if not ours(state, hold, review=True):
            return False
        updated = dict(hold)
        updated.setdefault('original_holder', state['holder'])
        updated['holder'] = f"{state['holder']}-hardware-review"
        updated['management_loss_latched'] = True
        updated['management_loss_phase'] = phase
        updated['management_loss_reason'] = str(reason)
        if details is not None:
            updated['management_loss_details'] = details
        updated['rule'] = (
            'Management or guard telemetry loss latched; retain hold for manual review '
            'and forbid automatic restoration mutations'
        )
        write_hold_atomic(hold_path, updated)
        return True


def active_markers(state):
    return [p for p in state.get('activity_markers', []) if Path(p).exists()]


def require_no_active_jobs(state):
    active = active_markers(state)
    if active:
        raise RuntimeError(f'Unresolved owned job markers block restoration: {active}')


def stale(hold, now=None):
    now = time.time() if now is None else now
    stamp = dt.datetime.fromisoformat(hold['heartbeat'].replace('Z', '+00:00')).timestamp()
    return now - stamp > 900


def checked(command, cwd, output=None):
    try:
        result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired as error:
        if output is not None and error.stdout:
            text = error.stdout if isinstance(error.stdout, str) else error.stdout.decode(errors='replace')
            output.write_text(text)
        raise ManagementLossError(f'{command}: timed out after {error.timeout}s') from error
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


def validate_restored_snapshot(state, snapshot):
    rows = snapshot['nodes']
    if set(rows) != set(state['restore_nodes']):
        raise RuntimeError('Restore snapshot has a missing or unexpected rank')
    for name, row in rows.items():
        if row['checkout'] != state['restore_commit'] or row['dirty']:
            raise RuntimeError(f'{name}: original clean checkout is not restored')
        if not row['running']:
            raise RuntimeError(f'{name}: original service is not running')
        if row['image_id'] != state['restore_image_id'] or row['image_tag'] != state['restore_image_tag']:
            raise RuntimeError(f'{name}: original image is not restored')
        if row['profile'] != state['restore_profile'] or not row['origin_matches']:
            raise RuntimeError(f'{name}: original profile/repository is not restored')
    if not snapshot['api_ready']:
        raise RuntimeError('Original API is not ready')


def verify_restored(state, controller, output):
    # Load the captured controller's own recipe and SSH helpers. doctor --live
    # alone permits missing containers as warnings and does not verify checkouts.
    script = r"""
import argparse, json, sys
from spark_ds41f.config import configuration, head_node, run_ssh
from spark_ds41f.cluster import remote_repository_state, normalized_repository_url
state = json.loads(sys.argv[1])
cluster, nodes, _ = configuration(argparse.Namespace(cluster_config=state['restore_profile']))
record = {'nodes': {}}
for node in nodes['nodes']:
    try:
        revision, origin, dirty = remote_repository_state(cluster, nodes, node)
    except Exception as error:
        raise RuntimeError(f"MANAGEMENT_LOSS: {node['name']}: {error}") from error
    inspected = run_ssh(nodes, node, 'docker', 'inspect', cluster['container']['name'])
    if inspected.returncode:
        detail = (inspected.stdout + '\n' + inspected.stderr).strip()
        if inspected.returncode == 255 or 'no such' not in detail.lower():
            raise RuntimeError(
                f"MANAGEMENT_LOSS: {node['name']}: docker inspect failed "
                f"(exit {inspected.returncode}): {detail}"
            )
        raise RuntimeError(f"{node['name']}: original container is absent")
    container = json.loads(inspected.stdout)[0]
    record['nodes'][node['name']] = {
        'checkout': revision, 'dirty': dirty, 'running': container['State']['Running'],
        'image_id': container['Image'], 'image_tag': container['Config']['Image'],
        'profile': (container['Config'].get('Labels') or {}).get('spark.cluster-config'),
        'origin_matches': normalized_repository_url(origin) == normalized_repository_url(cluster['deployment']['repository'])}
ready = run_ssh(nodes, head_node(nodes), 'curl', '--fail', '--silent', '--show-error', '--max-time', '5', cluster['deployment']['ready_url'])
if ready.returncode == 255:
    raise RuntimeError(f"MANAGEMENT_LOSS: head ready probe SSH failed: {ready.stderr.strip()}")
record['api_ready'] = ready.returncode == 0
print(json.dumps(record))
"""
    snapshot = json.loads(checked([sys.executable, '-c', script, json.dumps(state)], controller, output))
    validate_restored_snapshot(state, snapshot)


def restore(state, hold_path, output):
    controller = Path(state['restore_controller'])
    output.mkdir(parents=True, exist_ok=False)
    def own(for_release=False):
        hold = read_hold(hold_path)
        if not ours(state, hold, review=not for_release):
            raise RuntimeError('Window ownership changed; no further action')
        if hold.get('management_loss_latched') or hold.get('holder') != state['holder']:
            raise RuntimeError('Sticky management/telemetry loss latch is active; no restoration mutation')
        return hold
    own()
    require_no_active_jobs(state)
    try:
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
        already_restored = False
        try:
            checked([*base, 'doctor', '--live'], controller, output / 'doctor-before.txt')
            verify_restored(state, controller, output / 'snapshot-before.json')
            already_restored = True
        except Exception as error:
            if is_management_loss(error):
                latch_management_loss(state, hold_path, error, phase='initial-live-check')
                raise
        if not already_restored:
            require_no_active_jobs(state)
            steps = [('stop', []), ('sync', ['--revision', state['restore_commit']]), ('start', ['--replace'])]
            for action, extra in steps:
                own()
                require_no_active_jobs(state)
                argv = [*base, 'cluster', action, *extra]
                plan = checked(argv, controller, output / f'{action}-plan.txt')
                verify_plan(action, plan)
                own()
                require_no_active_jobs(state)
                checked([*argv, '--apply'], controller, output / f'{action}-apply.txt')
            own()
            checked([*base, 'doctor', '--live'], controller, output / 'doctor-live.txt')
        own()
        verify_restored(state, controller, output / 'snapshot-final.json')
    except Exception as error:
        if is_management_loss(error):
            latch_management_loss(state, hold_path, error, phase='restore')
        raise
    # Both the original commit and profile have been restored. Revalidate exact
    # ownership and unresolved job markers under the hold-mutation lock right
    # before unlinking so a foreign replacement is never removed or modified.
    with hold_mutation_lock(hold_path):
        require_no_active_jobs(state)
        own(for_release=True)
        (output / 'restored.json').write_text(json.dumps({
            'window': state['window'], 'commit': state['restore_commit'],
            'profile': state['restore_profile'], 'image': actual, 'restored': True}, indent=2) + '\n')
        own(for_release=True)
        hold_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=Path)
    parser.add_argument('--restore-now', action='store_true')
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    hold_path = Path(state['hold_path'])
    if not args.restore_now and ours(state, read_hold(hold_path)):
        args.state.with_suffix('.watchdog-ready.json').write_text(json.dumps({
            'window': state['window'], 'pid': os.getpid(), 'ready': True}) + '\n')
        print(f'Watchdog ready for {state["window"]}', flush=True)
    while True:
        hold = read_hold(hold_path)
        if not ours(state, hold, review=True):
            return 0
        if hold.get('management_loss_latched') or hold.get('holder') != state['holder']:
            print('Sticky management/telemetry loss latch active; hold retained for review', flush=True)
            return 1
        if args.restore_now or stale(hold):
            with Path(state['operation_lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                current = read_hold(hold_path)
                if not ours(state, current, review=True):
                    return 0
                if current.get('management_loss_latched') or current.get('holder') != state['holder']:
                    return 1
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
