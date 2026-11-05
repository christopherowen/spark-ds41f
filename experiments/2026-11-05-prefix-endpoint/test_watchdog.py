import unittest
from unittest.mock import patch
import branch_watchdog as guard
import json
import tempfile
from pathlib import Path
from window_lease import require_window


class WatchdogTest(unittest.TestCase):
    def test_holder_prefix_does_not_grant_ownership(self):
        state = {'window': 'ours', 'holder': 'spark-lab: prefix-endpoint'}
        self.assertFalse(guard.ours(state, {'window': 'theirs', 'holder': state['holder']}))
        self.assertFalse(guard.ours(state, {'window': 'ours', 'holder': 'spark-lab: someone else'}))
        self.assertFalse(guard.ours(state, None))
        self.assertTrue(guard.ours(state, dict(state)))

    def test_watchdog_requires_full_stale_interval(self):
        hold = {'heartbeat': '1970-01-01T00:00:00Z'}
        self.assertFalse(guard.stale(hold, 900))
        self.assertTrue(guard.stale(hold, 901))

    def test_changed_shape_is_never_applied(self):
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            guard.verify_plan('start', 'live: three ranks\nshape: CHANGES nodes\n')
        guard.verify_plan('start', 'live: stopped\nshape: unchanged\n')

    def test_refused_plan_is_never_applied(self):
        with self.assertRaisesRegex(RuntimeError, 'refusing'):
            guard.verify_plan('stop', 'live: four ranks\n--apply would refuse: partial stop\n')

    def test_foreign_hold_causes_no_command(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = {'window': 'ours', 'holder': 'prefix-endpoint', 'restore_controller': d}
            with patch.object(guard, 'checked') as command:
                with self.assertRaisesRegex(RuntimeError, 'ownership changed'):
                    guard.restore(state, root / 'absent-hold', root / 'out')
                command.assert_not_called()

    def test_request_without_owned_window_never_opens_network(self):
        import probe
        with patch.object(probe, "WINDOW_STATE", None), patch.object(probe.urllib.request, "urlopen") as opened:
            with self.assertRaisesRegex(RuntimeError, "owned window"):
                probe.post("http://unused", {})
            opened.assert_not_called()

    def test_stale_lease_refuses_future_requests(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = {'window': 'ours', 'holder': 'prefix-endpoint', 'hold_path': str(root / 'hold')}
            (root / 'state').write_text(json.dumps(state))
            (root / 'hold').write_text(json.dumps({**state, 'heartbeat': '1970-01-01T00:00:00Z'}))
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                require_window(root / 'state')

    def test_restore_requires_stop_sync_start_and_live_verification(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = {'window': 'ours', 'holder': 'prefix-endpoint', 'restore_controller': d,
                     'restore_commit': 'original', 'restore_image_id': 'image',
                     'restore_image_tag': 'saved-tag', 'restore_profile': 'saved-profile.json'}
            hold = root / 'hold'
            hold.write_text(json.dumps(state))
            calls = []
            def fake(argv, cwd, output=None):
                calls.append(argv)
                if argv[:2] == ['git', 'rev-parse']:
                    return 'original\n'
                if argv[0] == 'docker':
                    return 'image\n'
                return 'live: stopped\nshape: unchanged\n' if 'cluster' in argv else ''
            with patch.object(guard, 'checked', side_effect=fake):
                guard.restore(state, hold, root / 'out')
            self.assertFalse(hold.exists())
            mutations = [a[a.index('cluster') + 1] for a in calls if '--apply' in a]
            self.assertEqual(mutations, ['stop', 'sync', 'start'])
            self.assertEqual(calls[-1][-2:], ['doctor', '--live'])

    def test_failed_sync_keeps_the_hold_and_does_not_start(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = {'window': 'ours', 'holder': 'prefix-endpoint', 'restore_controller': d,
                     'restore_commit': 'original', 'restore_image_id': 'image',
                     'restore_image_tag': 'saved-tag', 'restore_profile': 'saved-profile.json'}
            hold = root / 'hold'
            hold.write_text(json.dumps(state))
            calls = []
            def fake(argv, cwd, output=None):
                calls.append(argv)
                if argv[:2] == ['git', 'rev-parse']:
                    return 'original\n'
                if argv[0] == 'docker':
                    return 'image\n'
                if 'sync' in argv and '--apply' in argv:
                    raise RuntimeError('sync failed')
                return 'live: stopped\nshape: unchanged\n' if 'cluster' in argv else ''
            with patch.object(guard, 'checked', side_effect=fake):
                with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                    guard.restore(state, hold, root / 'out')
            self.assertTrue(hold.exists())
            self.assertFalse(any('start' in argv for argv in calls))


if __name__ == '__main__':
    unittest.main()
