#!/usr/bin/env python3
"""Draft C46 source staging; not exercised when added. Default is a printed plan.

Run only from a clean published private controller checkout on DGX1, outside both
managed checkouts. Initialize its ignored site maps node-locally from the actual
DGX1 serving checkout beforehand; this helper neither creates that checkout nor
copies maps. Staging uses the normal named sync, never starts/stops an engine,
and cannot repair an unexpected runtime change. Existing queue order still applies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab
import c46_policy as policy
import c46_window as lease
from c46_management import ManagementLatch

SCRIPT = str(Path(__file__).resolve())
SERVING = Path('/home/swank/projects/spark-ds41f')
BUILDER = Path('/home/swank/projects/spark-ds41f-copy-default')
SPARK = [sys.executable, str(ROOT / 'bin/spark'), '--cluster-config', policy.RESTORE_PROFILE]


def pending():
    return lab.REQUEST.exists() or lab.LEGACY_REQUEST.exists()


def legacy_watchdogs():
    """A leftover broad-prefix watchdog must not be allowed to adopt this lease."""
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if b'watchdog' in argv and any(Path(os.fsdecode(arg)).name == 'lab.py' for arg in argv if arg):
            found.append(int(directory.name))
    return found


def require_ours(window, holder, token, management):
    hold = lab.read_hold()
    if (not lease.matches(hold, window, holder, token) or management.latched
            or hold.get('management_loss_latched')):
        raise RuntimeError('exact staging lease unavailable; no source operation authorized')
    return hold


def local_identity(revision):
    if sys.platform != 'linux' or socket.gethostname().split('.')[0] != 'dgx1':
        raise RuntimeError('staging controller must run on DGX1')
    for managed in (SERVING, BUILDER):
        if ROOT.resolve().is_relative_to(managed.resolve()):
            raise RuntimeError('use a private controller outside the managed checkouts')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=30).strip()
    if git('rev-parse', 'HEAD') != revision or git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('private controller must be clean at the exact target revision')
    cluster = lab.spark.read_cluster(policy.RESTORE_PROFILE)
    if Path(lab.spark.repository_path(cluster)) != SERVING:
        raise RuntimeError('named restoration profile does not select the managed serving checkout')
    maps = {cluster.get('nodes_config', 'config/nodes.json'), 'config/site.json'}
    if (SERVING / 'config/nodes.json').exists():
        maps.add('config/nodes.json')
    identities = {}
    for relative in sorted(maps):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or '.example.' in path.name:
            raise RuntimeError('actual repository-relative local site maps are required')
        actual, private = SERVING / path, ROOT / path
        if not actual.is_file() or not private.is_file() or actual.read_bytes() != private.read_bytes():
            raise RuntimeError('initialize matching actual site map node-locally: ' + relative)
        identities[relative] = hashlib.sha256(private.read_bytes()).hexdigest()
    return {'root': str(ROOT), 'commit': revision,
            'git_common_dir': git('rev-parse', '--path-format=absolute', '--git-common-dir'),
            'site_map_sha256': identities}


def sources(management, revision=None):
    """Before staging, different clean published revisions are intentional evidence."""
    result = {}
    expected = lab.spark.read_cluster(policy.RESTORE_PROFILE)['deployment']['repository']
    for host in policy.NODES:
        directory = str(BUILDER if host == 'dgx4' else SERVING)
        code = ("import json,subprocess;d=" + repr(directory) + ";"
                "g=lambda *a:subprocess.check_output(['git','-C',d,*a],text=True,timeout=30).strip();"
                "print(json.dumps({'commit':g('rev-parse','HEAD'),'dirty':g('status','--porcelain','--untracked-files=all'),"
                "'origin':g('remote','get-url','origin')}))")
        row = json.loads(management.ssh(host, ['python3', '-c', code], 'staging-source-identity', timeout=100))
        origin = row.pop('origin')
        row['origin_matches_authority'] = (lab.spark.normalized_repository_url(origin)
                                            == lab.spark.normalized_repository_url(expected))
        if (not re.fullmatch('[0-9a-f]{40}', row.get('commit', '')) or row.get('dirty')
                or not row['origin_matches_authority'] or revision and row['commit'] != revision):
            raise RuntimeError(host + ': clean configured source identity does not match')
        result[host] = row
    return result


def publication_guard(before, revision, management):
    """Reuse the lab guard; exempt only the exact observed clean pre-sync drift."""
    original_ssh, original_process = lab.spark.run_ssh, lab.subprocess
    class BoundedPublication:
        @staticmethod
        def run(*args, **kwargs):
            kwargs.setdefault('timeout', 60)
            value = subprocess.run(*args, **kwargs)
            if 'fetch' in args[0] and value.returncode:
                raise RuntimeError('publication fetch failed; stale origin state is insufficient')
            return value
    def read(nodes, node, *argv):
        text = management.ssh(lab.spark.ssh_target(nodes, node), list(argv), 'staging-publication')
        if node['name'] not in before or text.strip() != before[node['name']]['commit']:
            raise RuntimeError('checkout changed during the publication guard')
        return subprocess.CompletedProcess(argv, 0, text, '')
    try:
        lab.spark.run_ssh, lab.subprocess = read, BoundedPublication
        problems = lab.published_problems()
    finally:
        lab.spark.run_ssh, lab.subprocess = original_ssh, original_process
    allowed = {f'{host} checkout {row["commit"][:8]} differs from {revision[:8]}'
               for host, row in before.items() if host in ('dgx2', 'dgx3') and row['commit'] != revision}
    if set(problems) - allowed:
        raise RuntimeError('publication guard failed: ' + '; '.join(sorted(set(problems) - allowed)))
    for row in before.values():
        subprocess.run(['git', 'merge-base', '--is-ancestor', row['commit'], 'origin/main'],
                       cwd=ROOT, check=True, timeout=30)
    return {'problems': problems, 'permitted_clean_source_drift': sorted(allowed)}


def claims(management, token=None, window=None):
    for host in policy.NODES:
        records = json.loads(management.ssh(host, ['python3', '-c', policy.CLAIM_PROBE], 'staging-local-claims'))
        if token is None and any(value is not None for value in records.values()):
            raise RuntimeError(host + ': an existing local claim must be reconciled first')
        if token is not None:
            policy.validate_local_records(records, token, window)


def image_tags(management):
    # Normal sync also reconciles the named profile's tag. Require the already
    # installed immutable C36 image everywhere so staging does not copy a new one.
    tag = lab.spark.read_cluster(policy.RESTORE_PROFILE)['container']['image']
    values = {host: management.ssh(host, ['docker', 'image', 'inspect', '--format', '{{.Id}}', tag],
                                   'staging-existing-image').strip() for host in policy.NODES[:3]}
    if any(value != policy.RESTORE_IMAGE for value in values.values()):
        raise RuntimeError('named sync image differs from the unchanged C36 runtime; reconcile first')
    return values


def review(window, holder, token, management, error):
    if lease.matches(lab.read_hold(), window, holder, token, review=True):
        if management.latched:
            lease.hardware_review(window, holder, token, management)
        else:
            lease.update(window, holder, token, lambda h: h.update(
                holder='candidate-source-staging-review', staging_error=str(error),
                rule='Source staging requires review; no automatic restart, rollback or new work'), review=True)


def close(window, holder, token, out, management, watchdog=False):
    """Verify unchanged production; never repair it or call the generic window close."""
    with lab.HOLD.with_suffix('.c46-restore.lock').open('a+b') as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        hold = require_ours(window, holder, token, management)
        if lease._alive_restore({'c46_restore_process': hold.get('c46_staging_process', {})}):
            raise RuntimeError('exact source operation is still alive; keep its lease')
        if (not lease.verify_restored(out, management, 'staging-close')
                or lease._ranks(management) != hold['staging_initial_identity']):
            raise RuntimeError('C36 identity changed during staging; no automatic restoration launch')
        local_identity(hold['execution_commit'])
        lab.write_json_atomic(out / 'staging-close-sources.json', sources(management))
        lab.write_json_atomic(out / 'staging-close-image-tags.json', image_tags(management))
        with lease.mutation_lock():
            hold = require_ours(window, holder, token, management)
            if not watchdog:
                lease._stop_watchdog(hold)
            require_ours(window, holder, token, management)
            lab.HOLD.unlink()


def run_sync(argv, out, window, holder, token, management):
    require_ours(window, holder, token, management)
    path = out / 'sync-apply.log'
    with path.open('x') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        stat = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()
        job = {'pid': child.pid, 'start_ticks': int(stat[19]), 'pgid': os.getpgid(child.pid), 'argv': argv}
        lease.update(window, holder, token, lambda h: h.update(c46_staging_process=job))
        try:
            code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            management.latch('management_metadata_timeout', 'source-sync', detail='named source sync timed out')
            if child.poll() is None and os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            raise
    management.observe_file(path, 'source-sync')
    if code:
        raise RuntimeError('named source sync failed; original sync log retained')


def watchdog(args):
    management = ManagementLatch()
    while True:
        hold = lab.read_hold()
        if (not lease.matches(hold, args.window, args.holder, args.owner_token)
                or hold.get('execution_commit') != args.revision):
            return 0
        if hold.get('management_loss_latched'):
            return 1
        if time.time() - lab.parse_time(hold['heartbeat']) > lab.HEARTBEAT_MAX_AGE:
            out = ROOT / 'results/private/lab' / ('c46-staging-stale-' + args.owner_token)
            out.mkdir(exist_ok=False)
            try:
                close(args.window, args.holder, args.owner_token, out, management, watchdog=True)
                return 0
            except BaseException as error:
                review(args.window, args.holder, args.owner_token, management, error)
                lab.write_json_atomic(out / 'receipt.json', {'error': repr(error), **management.snapshot()})
                return 1
        time.sleep(15)


def apply(args):
    lab.PRODUCTION_CONFIG = policy.RESTORE_PROFILE
    controller = local_identity(args.revision)
    if lab.read_hold() or pending() or legacy_watchdogs():
        raise RuntimeError('shared hold/request exists; existing queue order is unchanged')
    token, management = uuid.uuid4().hex, ManagementLatch()
    out = ROOT / 'results/private/lab' / ('c46-source-staging-' + token)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'started': lab.now(), 'controller': controller, 'target_commit': args.revision,
               'source_only': True, 'serving_restart_authorized_here': False}
    window = holder = None
    stop, heartbeat_errors, heartbeat_thread = threading.Event(), [], None
    def heartbeat():
        while not stop.wait(15):
            try:
                lease.beat(window, holder, token, 'C46 source staging only; C36 remains serving')
            except BaseException as error:
                heartbeat_errors.append(repr(error)); return
    try:
        claims(management)
        receipt['before_sources'] = sources(management)
        receipt['publication'] = publication_guard(receipt['before_sources'], args.revision, management)
        if not lease.verify_restored(out, management, 'staging-open') or not lab.idle_for(30):
            raise RuntimeError('named immutable C36 service is not live and idle')
        initial = lease._ranks(management)
        if management.latched or lab.read_hold() or pending() or legacy_watchdogs():
            raise RuntimeError('shared state or management changed before staging acquisition')
        hold = lab.new_hold(0, 'owner-authorized C46 source staging only; keep matching C36 serving')
        holder, window = hold['holder'], hold['window']
        hold.update(c46_owner_token=token, c46_holder=holder, execution_commit=args.revision,
                    staging_initial_identity=initial, resource_scope={
                        'owner': policy.OWNER, 'owner_token': token, 'token': window, 'window': window,
                        'holder': holder, 'nodes': list(policy.NODES),
                        'controller': policy.process_identity(os.getpid(), args.revision, str(Path(SCRIPT).relative_to(ROOT))),
                        'restore': {'profile': policy.RESTORE_PROFILE, 'commit': args.revision,
                                    'image': policy.RESTORE_IMAGE, 'nodes': policy.NODES[:3]}})
        hold['nodes'] = list(policy.NODES)
        lease.exclusive_create(lab.HOLD, hold)
        argv = [sys.executable, SCRIPT, '--watchdog', '--revision', args.revision,
                '--window', window, '--holder', holder, '--owner-token', token]
        with (out / 'watchdog.log').open('x') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        lease._WATCHDOGS[child.pid] = child
        identity = policy.process_identity(child.pid, args.revision, str(Path(SCRIPT).relative_to(ROOT)))
        def bind(h):
            h['c46_watchdog'] = {'pid': child.pid, 'argv': argv}
            h['resource_scope']['watchdog'] = identity
        lease.update(window, holder, token, bind)
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        claims(management, token, window)
        receipt['existing_image_tags'] = image_tags(management)
        # Both are normal named plans; no start --apply exists in this helper.
        start_log = out / 'start-plan.log'
        if lease._local(SPARK + ['cluster', 'start', '--replace'], start_log, management, 'staging-start-plan'):
            raise RuntimeError('named C36 start plan failed')
        plan = start_log.read_text()
        if 'live:    running:' not in plan or 'shape:   unchanged' not in plan:
            raise RuntimeError('live/shape lines do not preserve the existing TP3 service')
        sync = SPARK + ['cluster', 'sync', '--revision', args.revision]
        if lease._local(sync, out / 'sync-plan.log', management, 'staging-sync-plan'):
            raise RuntimeError('named sync plan failed')
        if pending():
            receipt['deferred_after_open'] = 'request received before source mutation'
            stop.set(); heartbeat_thread.join(timeout=25)
            close(window, holder, token, out, management)
            return
        if heartbeat_errors:
            raise RuntimeError('heartbeat failed before the source-staging job')
        claims(management, token, window)
        run_sync(sync + ['--apply'], out, window, holder, token, management)
        require_ours(window, holder, token, management)
        # Only our owned DGX4 worktree uses a direct Git update; serving uses sync above.
        code = ("import subprocess;d=" + repr(str(BUILDER)) + ";r=" + repr(args.revision) + ";"
                "g=lambda *a:subprocess.check_output(['git','-C',d,*a],text=True,timeout=90).strip();"
                "assert not g('status','--porcelain','--untracked-files=all'),'owned builder is dirty';"
                "u=g('remote','get-url','origin').rstrip('/').removesuffix('.git');"
                "assert u==" + repr(lab.spark.normalized_repository_url(
                    lab.spark.read_cluster(policy.RESTORE_PROFILE)['deployment']['repository'])) + ", 'owned builder origin changed';"
                "g('fetch','--quiet','--prune','origin');g('merge-base','--is-ancestor',r,'origin/main');"
                "g('checkout','--quiet','--detach',r);print(g('rev-parse','HEAD'))")
        receipt['builder_update'] = management.ssh('dgx4', ['python3', '-c', code], 'staging-builder-git', timeout=240).strip()
        receipt['after_sources'] = sources(management, args.revision)
        receipt['controller_after'] = local_identity(args.revision)
        if heartbeat_errors:
            raise RuntimeError('staging heartbeat failed: ' + '; '.join(heartbeat_errors))
        stop.set(); heartbeat_thread.join(timeout=25)
        close(window, holder, token, out, management)
        receipt['completed'] = True
    except BaseException as error:
        receipt['error'] = repr(error)
        if window:
            review(window, holder, token, management, error)
        raise
    finally:
        stop.set()
        receipt.update(ended=lab.now(), window=window, holder=holder, owner_token=token,
                       heartbeat_errors=heartbeat_errors, **management.snapshot())
        lab.write_json_atomic(out / 'receipt.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watchdog', action='store_true', help=argparse.SUPPRESS)
    for name in ('window', 'holder', 'owner-token'):
        parser.add_argument('--' + name, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('revision must be one full published commit identity')
    if args.watchdog:
        if not all((args.window, args.holder, args.owner_token)):
            parser.error('exact watchdog window, holder and owner token are required')
        lab.PRODUCTION_CONFIG = policy.RESTORE_PROFILE
        return watchdog(args)
    if not args.apply:
        print('Plan only: clean published private DGX1 controller; actual node-local maps; shared guards;')
        print('exact exclusive staging lease and bound watchdog; named C36 start/sync plans;')
        print('named cluster sync to ' + args.revision + '; owned DGX4 clean Git update;')
        print('verify unchanged all-rank C36 image/profile/containers, then release without restart.')
        return 0
    apply(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
