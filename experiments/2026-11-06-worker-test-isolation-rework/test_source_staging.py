"""Offline staging failure checks. Authored without execution; run in a bounded CPU slot."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_source as stage
from c46_management import ManagementLatch


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private, self.serving, self.builder = (self.root / n for n in ('private', 'serving', 'builder'))
        self.private.mkdir(); self.serving.mkdir(); self.builder.mkdir()
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for obj, name, value in ((stage, 'ROOT', self.private), (stage, 'SERVING', self.serving),
                                  (stage, 'BUILDER', self.builder), (stage.lab, 'HOLD', self.root / 'hold.json'),
                                  (stage, 'SCRIPT', str(self.private / 'stage_source.py')),
                                  (stage.lab, 'PRODUCTION_CONFIG', stage.policy.RESTORE_PROFILE),
                                  (stage.lab, 'LEGACY_HOLD', self.root / 'legacy-hold.json'),
                                  (stage.lab, 'REQUEST', self.root / 'request.json'),
                                  (stage.lab, 'LEGACY_REQUEST', self.root / 'legacy-request.json')):
            self.stack.enter_context(mock.patch.object(obj, name, value))
        self.commit, self.token, self.window = 'f' * 40, 'a' * 32, 'dgx1-staging'
        self.holder = 'spark-lab: exact source staging'
        self.rows = [{'host': h, 'id': str(i) * 64, 'image': stage.policy.RESTORE_IMAGE,
                      'running': True, 'started': 'unchanged', 'restarts': 0,
                      'profile': stage.policy.RESTORE_PROFILE}
                     for i, h in enumerate(stage.policy.NODES[:3], 1)]
        self.own = {'holder': self.holder, 'window': self.window, 'c46_owner_token': self.token,
                    'c46_holder': self.holder, 'execution_commit': self.commit,
                    'staging_initial_identity': self.rows}

    def install(self, value=None):
        stage.lab.HOLD.write_text(json.dumps(self.own if value is None else value))

    def test_atomic_acquisition_loses_race_without_replacing_foreign_hold(self):
        original_link = stage.lease.os.link
        foreign = {'holder': 'foreign', 'window': 'foreign-window'}
        def race(source, target):
            target.write_text(json.dumps(foreign))
            return original_link(source, target)
        with mock.patch.object(stage.lease.os, 'link', side_effect=race):
            with self.assertRaises(FileExistsError):
                stage.lease.exclusive_create(stage.lab.HOLD, self.own)
        self.assertEqual(json.loads(stage.lab.HOLD.read_text()), foreign)
        self.assertEqual(list(self.root.glob('.c46-hold-*')), [])

    def test_watchdog_cannot_follow_other_window_token_holder_or_commit(self):
        args = SimpleNamespace(window=self.window, holder=self.holder, owner_token=self.token, revision=self.commit)
        for changed in ({'window': 'other'}, {'c46_owner_token': 'other'},
                        {'holder': 'foreign'}, {'execution_commit': 'e' * 40}):
            self.install({**self.own, **changed})
            before = stage.lab.HOLD.read_bytes()
            with mock.patch.object(stage, 'close') as close, mock.patch.object(stage.time, 'sleep') as sleep:
                self.assertEqual(stage.watchdog(args), 0)
            close.assert_not_called(); sleep.assert_not_called()
            self.assertEqual(stage.lab.HOLD.read_bytes(), before)

    def test_leftover_generic_watchdog_is_identified_without_stopping_it(self):
        proc = self.root / 'proc'; proc.mkdir()
        for pid, script, mode in ((123, 'lab.py', 'watchdog'), (456, 'stage_source.py', '--watchdog')):
            directory = proc / str(pid); directory.mkdir()
            (directory / 'cmdline').write_bytes(b'python3\0' + script.encode() + b'\0' + mode.encode() + b'\0')
        real_path = Path
        with mock.patch.object(stage, 'Path', side_effect=lambda p: proc if p == '/proc' else real_path(p)), \
                mock.patch.object(stage.os, 'kill') as kill:
            self.assertEqual(stage.legacy_watchdogs(), [123])
        kill.assert_not_called()

    def test_live_source_operation_retains_lease_without_doctor_or_restart(self):
        self.install({**self.own, 'c46_staging_process': {'pid': 123, 'start_ticks': 456}})
        before = stage.lab.HOLD.read_bytes()
        with mock.patch.object(stage.lease, '_alive_restore', return_value=True), \
                mock.patch.object(stage.lease, 'verify_restored') as doctor, \
                mock.patch.object(stage.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                stage.close(self.window, self.holder, self.token, self.root, ManagementLatch())
        doctor.assert_not_called(); spawn.assert_not_called()
        self.assertEqual(stage.lab.HOLD.read_bytes(), before)

    def test_matching_c36_close_only_runs_live_doctor_and_stops_own_watchdog(self):
        self.install({**self.own, 'c46_watchdog': {'pid': 123}})
        child, foreign = mock.Mock(), mock.Mock()
        child.poll.return_value = None
        def doctor(argv, path, *_args):
            self.assertEqual(argv[-2:], ['doctor', '--live'])
            path.write_text('live cluster matches'); return 0
        with mock.patch.object(stage.lease, '_WATCHDOGS', {123: child, 456: foreign}), \
                mock.patch.object(stage.lease, '_alive_restore', return_value=False), \
                mock.patch.object(stage.lease, '_ranks', return_value=self.rows), \
                mock.patch.object(stage.lease, '_local', side_effect=doctor) as commands, \
                mock.patch.object(stage, 'local_identity'), mock.patch.object(stage, 'sources', return_value={}), \
                mock.patch.object(stage, 'image_tags', return_value={}), \
                mock.patch.object(stage.subprocess, 'Popen') as spawn:
            stage.close(self.window, self.holder, self.token, self.root, ManagementLatch())
        self.assertFalse(stage.lab.HOLD.exists())
        self.assertEqual(commands.call_count, 1)
        child.terminate.assert_called_once(); child.wait.assert_called_once_with(timeout=5)
        foreign.terminate.assert_not_called(); spawn.assert_not_called()

    def test_c36_change_or_foreign_replacement_prevents_release(self):
        for foreign_replace in (False, True):
            self.install()
            def verify(*_args):
                if foreign_replace:
                    self.install({'holder': 'foreign', 'window': 'other'})
                return True
            with mock.patch.object(stage.lease, '_alive_restore', return_value=False), \
                    mock.patch.object(stage.lease, 'verify_restored', side_effect=verify), \
                    mock.patch.object(stage.lease, '_ranks', return_value=[{**self.rows[0], 'id': 'changed'}]), \
                    mock.patch.object(stage.lease, '_stop_watchdog') as stop:
                with self.assertRaises(RuntimeError):
                    stage.close(self.window, self.holder, self.token, self.root, ManagementLatch())
            self.assertTrue(stage.lab.HOLD.exists()); stop.assert_not_called()
            if foreign_replace:
                self.assertEqual(json.loads(stage.lab.HOLD.read_text())['holder'], 'foreign')

    def test_actual_site_map_hash_mismatch_is_rejected_before_any_window(self):
        paths = ('config/nodes.actual.json', 'config/site.json')
        for root in (self.private, self.serving):
            for name in paths:
                path = root / name; path.parent.mkdir(exist_ok=True); path.write_text('actual-node-local')
        cluster = {'nodes_config': paths[0]}
        def git(argv, **_kwargs):
            if 'status' in argv: return ''
            return self.commit if argv[-1] == 'HEAD' else str(self.root / 'common')
        with mock.patch.object(stage.sys, 'platform', 'linux'), \
                mock.patch.object(stage.socket, 'gethostname', return_value='dgx1'), \
                mock.patch.object(stage.subprocess, 'check_output', side_effect=git), \
                mock.patch.object(stage.lab.spark, 'read_cluster', return_value=cluster), \
                mock.patch.object(stage.lab.spark, 'repository_path', return_value=str(self.serving)):
            observed = stage.local_identity(self.commit)
            self.assertEqual(observed['site_map_sha256'][paths[0]], hashlib.sha256(b'actual-node-local').hexdigest())
            (self.private / paths[0]).write_text('an-example-or-another-site')
            with self.assertRaisesRegex(RuntimeError, 'actual site map'):
                stage.local_identity(self.commit)
        self.assertFalse(stage.lab.HOLD.exists())

    def test_source_inspection_allows_clean_drift_but_rejects_dirty_source(self):
        origin = 'https://github.com/christopherowen/spark-ds41f'
        raw = {'commit': 'b' * 40, 'dirty': '', 'origin': origin}
        management = ManagementLatch()
        with mock.patch.object(stage.lab.spark, 'read_cluster', return_value={'deployment': {'repository': origin}}), \
                mock.patch.object(management, 'ssh', return_value=json.dumps(raw)):
            observed = stage.sources(management)
            self.assertEqual(len(observed), 4)
            with self.assertRaises(RuntimeError): stage.sources(management, self.commit)
        with mock.patch.object(stage.lab.spark, 'read_cluster', return_value={'deployment': {'repository': origin}}), \
                mock.patch.object(management, 'ssh', return_value=json.dumps({**raw, 'dirty': ' M tracked'})):
            with self.assertRaisesRegex(RuntimeError, 'clean configured source'):
                stage.sources(management)

    def test_publication_adapter_accepts_only_observed_published_drift(self):
        before = {host: {'commit': str(i) * 40} for i, host in enumerate(stage.policy.NODES, 1)}
        nodes = {'nodes': [{'name': h, 'head': h == 'dgx1'} for h in stage.policy.NODES[:3]]}
        management = ManagementLatch()
        calls, published = [], True
        def git(argv, **kwargs):
            calls.append((argv, kwargs))
            text = self.commit + '\n' if 'rev-parse' in argv else ''
            code = int(not published) if 'merge-base' in argv else 0
            return subprocess.CompletedProcess(argv, code, text, '')
        with mock.patch.object(stage.lab, 'ROOT', self.private), \
                mock.patch.object(stage.subprocess, 'run', side_effect=git), \
                mock.patch.object(stage.lab, 'production_configuration', return_value=({}, nodes, {})), \
                mock.patch.object(stage.lab.spark, 'repository_path', return_value=str(self.serving)), \
                mock.patch.object(stage.lab.spark, 'ssh_target', side_effect=lambda _nodes, node: node['name']), \
                mock.patch.object(management, 'ssh', side_effect=lambda host, *_args: before[host]['commit']):
            value = stage.publication_guard(before, self.commit, management)
            self.assertEqual(len(value['permitted_clean_source_drift']), 2)
            self.assertTrue(all(kwargs.get('cwd') == self.private or ['-C', str(self.private)] == argv[1:3]
                                for argv, kwargs in calls))
            self.assertTrue(all(kwargs['timeout'] <= 60 for _argv, kwargs in calls))
            published = False
            with self.assertRaisesRegex(RuntimeError, 'publication guard failed'):
                stage.publication_guard(before, self.commit, management)

    def test_request_before_mutation_defers_and_spawns_only_exact_private_watchdog(self):
        captured, spawned = {}, []
        child = mock.Mock(pid=987); child.poll.return_value = None
        def close(*_args): stage.lab.HOLD.unlink()
        def plan(_argv, path, *_args):
            path.write_text('live:    running: C36\nshape:   unchanged\n'); return 0
        # Save the actual exclusive primitive before replacing the adapter's alias.
        actual_create = stage.lease.exclusive_create
        def create(path, hold): captured.update(hold); actual_create(path, hold)
        with mock.patch.object(stage, 'local_identity', return_value={'commit': self.commit}), \
                mock.patch.object(stage, 'pending', side_effect=[False, False, True]), \
                mock.patch.object(stage, 'legacy_watchdogs', return_value=[]), \
                mock.patch.object(stage, 'claims'), mock.patch.object(stage, 'sources', return_value={}), \
                mock.patch.object(stage, 'publication_guard', return_value={}), \
                mock.patch.object(stage, 'image_tags', return_value={}), \
                mock.patch.object(stage.lease, 'verify_restored', return_value=True), \
                mock.patch.object(stage.lease, '_ranks', return_value=self.rows), \
                mock.patch.object(stage.lab, 'idle_for', return_value=True), \
                mock.patch.object(stage.lease, 'exclusive_create', side_effect=create), \
                mock.patch.object(stage.policy, 'process_identity', return_value={'pid': 987}), \
                mock.patch.object(stage.subprocess, 'Popen', side_effect=lambda argv, **_kwargs: spawned.append(argv) or child), \
                mock.patch.object(stage.threading, 'Thread'), \
                mock.patch.object(stage.lease, '_local', side_effect=plan), \
                mock.patch.object(stage, 'close', side_effect=close), \
                mock.patch.object(stage, 'run_sync') as sync, \
                mock.patch.object(stage.lease, '_WATCHDOGS', {}):
            stage.apply(SimpleNamespace(revision=self.commit))
        sync.assert_not_called(); self.assertFalse(stage.lab.HOLD.exists())
        self.assertEqual(len(spawned), 1)
        self.assertEqual(spawned[0][1], stage.SCRIPT)
        self.assertIn('--watchdog', spawned[0]); self.assertNotIn('lab.py', spawned[0])
        for option, value in (('--window', captured['window']), ('--holder', captured['holder']),
                              ('--owner-token', captured['c46_owner_token']), ('--revision', self.commit)):
            self.assertEqual(spawned[0][spawned[0].index(option) + 1], value)
        self.assertEqual(captured['nodes'], stage.policy.NODES)
        self.assertEqual(captured['resource_scope']['nodes'], stage.policy.NODES)
        self.assertEqual(captured['resource_scope']['holder'], captured['holder'])
        receipt = json.loads(next((self.private / 'results/private/lab').glob('*/receipt.json')).read_text())
        self.assertIn('deferred_after_open', receipt)


if __name__ == '__main__':
    unittest.main()
