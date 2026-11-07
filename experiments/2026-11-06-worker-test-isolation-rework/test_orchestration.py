"""Offline C46 ownership, no-duplicate restoration and sticky-loss controls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c46_window as window
import c46_phase as phase
from c46_management import ManagementLatch


class ManagementTests(unittest.TestCase):
    def test_transport_loss_stays_latched_after_successful_probe(self):
        guard = ManagementLatch()
        results = [subprocess.CompletedProcess([], 255, '', 'ssh: connect to host dgx4: Connection timed out'),
                   subprocess.CompletedProcess([], 0, '', '')]
        with mock.patch('c46_management.subprocess.run', side_effect=results):
            with self.assertRaises(subprocess.CalledProcessError): guard.ssh('dgx4', ['true'], 'read')
            guard.ssh('dgx4', ['true'], 'recovered', probe=True)
        self.assertTrue(guard.latched)
        self.assertEqual(guard.reads[-1]['exit'], 0)

    def test_timeout_is_loss_but_missing_or_malformed_metadata_is_not(self):
        guard = ManagementLatch()
        with mock.patch('c46_management.subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', 'cat: no such file')):
            with self.assertRaises(subprocess.CalledProcessError): guard.ssh('dgx4', ['cat', 'missing'], 'metadata')
        self.assertFalse(guard.latched)
        with mock.patch('c46_management.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '{broken', '')):
            with self.assertRaises(json.JSONDecodeError): json.loads(guard.ssh('dgx4', ['cat', 'bad'], 'metadata'))
        self.assertFalse(guard.latched)
        with mock.patch('c46_management.subprocess.run', side_effect=subprocess.TimeoutExpired(['ssh'], 20)):
            with self.assertRaises(subprocess.TimeoutExpired): guard.ssh('dgx4', ['true'], 'metadata')
        self.assertTrue(guard.latched)

    def test_guard_loss_is_seen_even_with_parent_exit_zero(self):
        guard = ManagementLatch()
        guard.observe_text("HALT: dgx2: memory telemetry stopped reporting", 'performance')
        self.assertTrue(guard.latched)
        self.assertEqual(guard.events[0]['category'], 'guard_telemetry_loss')
        thermal = ManagementLatch()
        thermal.observe_text('RuntimeError: dgx4: thermal telemetry stopped reporting', 'image-and-gate')
        self.assertTrue(thermal.latched)
        compact = ManagementLatch()
        compact.observe_conditions({'halted': ['dgx4: thermal telemetry invalid after recovery']}, 'native-image-guard', 'dgx4')
        self.assertTrue(compact.latched)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.path = self.root / 'hold.json'
        patch = mock.patch.object(window.lab, 'HOLD', self.path); patch.start(); self.addCleanup(patch.stop)
        self.token = 'a' * 32; self.holder = 'spark-lab: exact C46'; self.window = 'dgx1-example'; self.commit = 'b' * 40
        self.own = {'holder': self.holder, 'window': self.window, 'c46_owner_token': self.token,
                    'c46_holder': self.holder, 'execution_commit': self.commit}
        self.foreign = {'holder': 'spark-lab: foreign', 'window': 'other', 'c46_owner_token': 'c' * 32}

    def test_exclusive_install_never_replaces_foreign_hold(self):
        self.path.write_text(json.dumps(self.foreign)); before = self.path.read_bytes()
        with self.assertRaises(FileExistsError): window.exclusive_create(self.path, self.own)
        self.assertEqual(self.path.read_bytes(), before)

    def test_foreign_close_and_old_watchdog_do_nothing(self):
        self.path.write_text(json.dumps(self.foreign))
        guard = ManagementLatch()
        with self.assertRaises(RuntimeError):
            window.close_window(self.window, self.holder, self.token, self.commit, self.root / 'close', guard)
        self.assertEqual(guard.reads, [])
        args = SimpleNamespace(window=self.window, holder=self.holder, owner_token=self.token,
                               commit=self.commit, restore_profile=window.policy.RESTORE_PROFILE)
        with mock.patch.object(window.lab, 'PRODUCTION_CONFIG', window.lab.PRODUCTION_CONFIG):
            self.assertEqual(window.watchdog_main(args), 0)
        self.assertEqual(json.loads(self.path.read_text()), self.foreign)

    def test_sticky_review_forbids_restoration_before_any_probe(self):
        self.path.write_text(json.dumps(self.own)); guard = ManagementLatch(); guard.latch('management', 'original')
        with self.assertRaises(RuntimeError):
            window.close_window(self.window, self.holder, self.token, self.commit, self.root / 'close', guard)
        self.assertEqual(guard.reads, [])
        self.assertTrue(json.loads(self.path.read_text())['management_loss_latched'])

    def test_exact_digest_is_required_even_when_doctor_matches(self):
        rows = [{'host': h, 'id': h, 'image': 'sha256:' + 'f' * 64, 'running': True,
                 'profile': window.policy.RESTORE_PROFILE} for h in ('dgx1', 'dgx2', 'dgx3')]
        def doctor(_argv, path, *_args): path.write_text('live cluster matches'); return 0
        with mock.patch.object(window, '_ranks', return_value=rows), mock.patch.object(window, '_local', side_effect=doctor):
            self.assertFalse(window.verify_restored(self.root, ManagementLatch(), 'test'))

    def test_foreign_replacement_during_verification_is_preserved(self):
        self.path.write_text(json.dumps(self.own)); guard = ManagementLatch()
        def replace(*_args): self.path.write_text(json.dumps(self.foreign)); return True
        with mock.patch.object(guard, 'ssh', return_value=''), mock.patch.object(window, 'clean_sources', return_value={}), \
                mock.patch.object(window, 'verify_restored', side_effect=replace):
            with self.assertRaises(RuntimeError):
                window.close_window(self.window, self.holder, self.token, self.commit, self.root / 'close', guard)
        self.assertEqual(json.loads(self.path.read_text()), self.foreign)

    def test_prior_restore_attempt_cannot_launch_again(self):
        self.path.write_text(json.dumps({**self.own, 'c46_restore_process': {'pid': 999, 'start_ticks': 1}}))
        guard = ManagementLatch()
        with mock.patch.object(guard, 'ssh', return_value=''), mock.patch.object(window, 'clean_sources', return_value={}), \
                mock.patch.object(window, 'verify_restored', return_value=False), mock.patch.object(window, '_alive_restore', return_value=False), \
                mock.patch.object(window, '_local') as launch:
            with self.assertRaises(RuntimeError):
                window.close_window(self.window, self.holder, self.token, self.commit, self.root / 'close', guard)
        launch.assert_not_called()


class PhaseTests(unittest.TestCase):
    def test_functional_guard_uses_existing_watcher_and_only_own_group(self):
        nodes = {'nodes': [{'name': 'dgx1'}, {'name': 'dgx2'}, {'name': 'dgx3'}]}
        watcher = SimpleNamespace(wait_ready=mock.Mock(), check=mock.Mock(),
                                  thermal_summary=mock.Mock(return_value={}), close=mock.Mock(return_value={}))
        factory = mock.Mock(return_value=watcher)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(phase.os, 'getpid', return_value=123), \
                mock.patch.object(phase.os, 'getpgid', return_value=123), mock.patch.object(phase.os, 'killpg') as stop, \
                mock.patch.object(phase.threading, 'Thread'):
            management = ManagementLatch()
            guard = phase.FunctionalGuard(nodes, 5, management, temporary, factory)
            guard.start()
            factory.assert_called_once_with(nodes, 5, guard.halt)
            watcher.wait_ready.assert_called_once()
            guard.halt('dgx1: thermal telemetry stopped reporting')
            guard.halt('second callback must not signal cleanup twice')
            stop.assert_called_once_with(123, phase.signal.SIGINT)
            self.assertEqual(len(guard.reasons), 2)
            self.assertTrue(management.latched)
            guard.close(); watcher.close.assert_called_once()

    def test_functional_guard_refuses_shared_group_before_watcher(self):
        factory = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(phase.os, 'getpid', return_value=123), \
                mock.patch.object(phase.os, 'getpgid', return_value=999), mock.patch.object(phase.os, 'killpg') as stop:
            guard = phase.FunctionalGuard({'nodes': []}, 5, ManagementLatch(), temporary, factory)
            with self.assertRaises(RuntimeError): guard.start()
        factory.assert_not_called(); stop.assert_not_called()

    def test_child_defers_failure_restoration_and_cannot_open_new_lease(self):
        args = SimpleNamespace(window='dgx1-test', holder='spark-lab: own', owner_token='a' * 32)
        original_close, original_open = phase.lab.window_close, phase.lab.window_open
        def execute(*_args, **kwargs):
            self.assertEqual(phase.os.environ['SPARK_DS41F_STRICT_TELEMETRY'], '1')
            self.assertFalse(kwargs['close_on_failure'])
            self.assertFalse(phase.lab.window_close('failed'))
            with self.assertRaises(RuntimeError): phase.lab.window_open(0, 'other')
            return 1
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(phase.policy, 'assert_owned'), \
                mock.patch.object(phase.lab, 'execute', side_effect=execute):
            self.assertEqual(phase.run_phase({}, args, Path(temporary) / 'phase'), 1)
        self.assertIs(phase.lab.window_close, original_close)
        self.assertIs(phase.lab.window_open, original_open)


if __name__ == '__main__':
    unittest.main()
