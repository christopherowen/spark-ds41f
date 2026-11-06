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
                     'restore_commit': 'original', 'restore_image_id': 'image', 'restore_nodes': ['rank0'],
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
                if argv[-2:] == ['doctor', '--live'] and not any('start' in call for call in calls):
                    raise RuntimeError('candidate is live')
                return 'live: stopped\nshape: unchanged\n' if 'cluster' in argv else ''
            with patch.object(guard, 'checked', side_effect=fake), patch.object(guard, 'verify_restored'):
                guard.restore(state, hold, root / 'out')
            self.assertFalse(hold.exists())
            mutations = [a[a.index('cluster') + 1] for a in calls if '--apply' in a]
            self.assertEqual(mutations, ['stop', 'sync', 'start'])
            self.assertEqual(calls[-1][-2:], ['doctor', '--live'])

    def test_failed_sync_keeps_the_hold_and_does_not_start(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = {'window': 'ours', 'holder': 'prefix-endpoint', 'restore_controller': d,
                     'restore_commit': 'original', 'restore_image_id': 'image', 'restore_nodes': ['rank0'],
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
                if argv[-2:] == ['doctor', '--live'] and not any('start' in call for call in calls):
                    raise RuntimeError('candidate is live')
                if 'sync' in argv and '--apply' in argv:
                    raise RuntimeError('sync failed')
                return 'live: stopped\nshape: unchanged\n' if 'cluster' in argv else ''
            with patch.object(guard, 'checked', side_effect=fake), patch.object(guard, 'verify_restored'):
                with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                    guard.restore(state, hold, root / 'out')
            self.assertTrue(hold.exists())
            self.assertFalse(any('start' in argv for argv in calls))


    def test_restored_snapshot_rejects_missing_stopped_or_changed_ranks(self):
        import copy
        state = {'restore_nodes': ['rank0', 'rank1'], 'restore_commit': 'original',
                 'restore_image_id': 'image', 'restore_image_tag': 'tag', 'restore_profile': 'profile'}
        row = {'checkout': 'original', 'dirty': '', 'running': True, 'image_id': 'image',
               'image_tag': 'tag', 'profile': 'profile', 'origin_matches': True}
        good = {'nodes': {'rank0': dict(row), 'rank1': dict(row)}, 'api_ready': True}
        guard.validate_restored_snapshot(state, good)
        bad = copy.deepcopy(good)
        del bad['nodes']['rank1']
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            guard.validate_restored_snapshot(state, bad)
        for field, value in [('running', False), ('checkout', 'candidate'), ('dirty', ' M file'),
                             ('image_id', 'candidate'), ('profile', 'candidate'), ('origin_matches', False)]:
            bad = copy.deepcopy(good)
            bad['nodes']['rank1'][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                guard.validate_restored_snapshot(state, bad)
        bad = copy.deepcopy(good)
        bad['api_ready'] = False
        with self.assertRaisesRegex(RuntimeError, 'API'):
            guard.validate_restored_snapshot(state, bad)

    def test_doctor_success_cannot_release_an_unrestored_cluster(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = {'window': 'ours', 'holder': 'prefix-endpoint', 'restore_controller': d,
                     'restore_commit': 'original', 'restore_image_id': 'image', 'restore_nodes': ['rank0'],
                     'restore_image_tag': 'tag', 'restore_profile': 'profile'}
            hold = root / 'hold'
            hold.write_text(json.dumps(state))
            calls = []
            def fake(argv, cwd, output=None):
                calls.append(argv)
                if argv[:2] == ['git', 'rev-parse']:
                    return 'original'
                if argv[0] == 'docker':
                    return 'image'
                return 'live: stopped\nshape: unchanged\n' if 'cluster' in argv else ''
            with patch.object(guard, 'checked', side_effect=fake), patch.object(guard, 'verify_restored', side_effect=RuntimeError('missing rank')):
                with self.assertRaisesRegex(RuntimeError, 'missing rank'):
                    guard.restore(state, hold, root / 'out')
            self.assertTrue(hold.exists())
            self.assertEqual([a[a.index('cluster') + 1] for a in calls if '--apply' in a], ['stop', 'sync', 'start'])


if __name__ == '__main__':
    unittest.main()
