import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import serving_gate
from serving_verdicts import verdict
from record_requests import RequestArchive
import urllib.request


class ServingGuards(unittest.TestCase):
    def sampled_rows(self):
        digest = 'a' * 64
        return [{'scenario': name, 'differing': [], 'sha256': digest} for name in ('c8', 'again')] + [{
            'summary': 'seeded temperature-1 digest', 'identical': True, 'tokens': 192,
            'seed': 20261024, 'scenarios': {'c8': {'differing': []}, 'again': {'differing': []}},
            'reference_sha256': digest}]

    def result(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'output.txt'
            p.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            return verdict(p, 'sampled')

    def test_zero_exit_negative_summary_is_not_a_pass(self):
        rows = self.sampled_rows()
        self.assertTrue(self.result(rows)['passed'])
        rows[-1]['identical'] = False
        rows[-1]['scenarios']['c8']['differing'] = [3]
        self.assertFalse(self.result(rows)['passed'])

    def test_summary_cannot_hide_missing_or_different_raw_outputs(self):
        rows = self.sampled_rows()
        self.assertFalse(self.result(rows[1:])['passed'])
        rows[0]['sha256'] = 'b' * 64
        self.assertFalse(self.result(rows)['passed'])

    def test_concurrent_archive_preserves_exact_bodies(self):
        with tempfile.TemporaryDirectory() as d:
            archive = RequestArchive(Path(d) / 'requests')
            bodies = [json.dumps({'prompt': f'input {i}', 'seed': i}).encode() for i in range(16)]
            threads = [threading.Thread(target=archive.add, args=(urllib.request.Request(
                'http://127.0.0.1:8000/v1/completions', data=body),)) for body in bodies]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            records = json.loads((archive.directory / 'manifest.json').read_text())['requests']
            self.assertEqual(len(records), 16)
            self.assertEqual({(archive.directory / row['body_file']).read_bytes() for row in records}, set(bodies))

    def test_monitor_failure_terminates_client_before_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            c = types.SimpleNamespace(stamp=lambda: 'test')
            with patch.object(serving_gate, 'state', side_effect=[{}, RuntimeError('thermal guard')]):
                with self.assertRaisesRegex(RuntimeError, 'thermal guard'):
                    serving_gate.run(c, 'token', [sys.executable, '-c', 'import time; time.sleep(20)'],
                                     out, 'client', floor_gib=3, timeout=30)
            receipt = json.loads((out / 'client-receipt.json').read_text())
            self.assertIn('thermal guard', receipt['error'])
            self.assertLess(receipt['terminated_exit'], 0)

    def test_foreign_lease_is_rejected(self):
        row = {'hold': {'holder': 'foreign', 'token': 'token', 'nodes': ['sp5','sp6','sp7'], 'phase': 'serving'},
               'thermal_zones_c': {'zone': 40}, 'gpu_temperature_c': 40, 'host_available_bytes': 10*2**30}
        c = types.SimpleNamespace(global_scope=lambda: None, HOLDER='ced-sp567-native-correctness',
            NAMES=['sp5','sp6','sp7'], recovery=types.SimpleNamespace(each=lambda code: {'sp5': row}))
        with self.assertRaisesRegex(AssertionError, 'lease changed'):
            serving_gate.state(c, 'token', 3)

    def test_failed_probe_retains_logs_before_original_error_returns(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(serving_gate, 'collect_logs') as collect:
                with self.assertRaisesRegex(ValueError, 'probe failure'):
                    with serving_gate.retained_engine_logs(None, Path(d), {}):
                        raise ValueError('probe failure')
                collect.assert_called_once()

    def test_log_failure_does_not_mask_failed_probe(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(serving_gate, 'collect_logs', side_effect=OSError('log unavailable')):
                with self.assertRaisesRegex(ValueError, 'probe failure'):
                    with serving_gate.retained_engine_logs(None, Path(d), {}):
                        raise ValueError('probe failure')
            self.assertIn('log unavailable', (Path(d) / 'engine-log-collection-error.txt').read_text())

    def test_successful_probe_still_fails_if_logs_are_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(serving_gate, 'collect_logs', side_effect=OSError('log unavailable')):
                with self.assertRaisesRegex(OSError, 'log unavailable'):
                    with serving_gate.retained_engine_logs(None, Path(d), {}):
                        pass

    def test_first_rank_fault_does_not_discard_other_rank_logs(self):
        nodes = ['node5', 'node6', 'node7']
        c = types.SimpleNamespace(NAMES=['sp5', 'sp6', 'sp7'],
                                  recovery=types.SimpleNamespace(NODES=nodes))
        before = {node: {'id': node + '-container'} for node in nodes}
        calls = []
        def run(command, *, stdout, **kwargs):
            calls.append(command[-1])
            stdout.write(b'CUDA error: retained failure' if len(calls) == 1 else b'healthy worker')
            return types.SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(serving_gate.subprocess, 'run', side_effect=run):
                with self.assertRaises(AssertionError):
                    serving_gate.collect_logs(c, Path(d), before)
            records = json.loads((Path(d) / 'engine-logs.json').read_text())
            self.assertEqual(len(calls), 3)
            self.assertIn('engine fault', records['sp5']['issues'])
            self.assertFalse(records['sp6']['issues'])
            self.assertFalse(records['sp7']['issues'])


if __name__ == '__main__':
    unittest.main()
