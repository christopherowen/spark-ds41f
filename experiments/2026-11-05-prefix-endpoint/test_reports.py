import json
import tempfile
import unittest
from pathlib import Path
from tables_arms import report
from compare import paired


class ReportTest(unittest.TestCase):
    def fixture(self, root):
        for label, seconds in [('boot-a', 1), ('boot-b', 3)]:
            for i in range(3):
                p = root / label / f'rep-{i}'
                p.mkdir(parents=True)
                p.with_suffix('.conditions.json').write_text(json.dumps({'passed': True}))
                (p / 'summary.json').write_text(json.dumps({'passed': True}))
                (p / 'workload.json').write_text(json.dumps({'prompt_hashes': ['same']}))
                rows = [{'label': f'{kind}-{turn}', 'ttft_s': seconds,
                         'cached_tokens': 0, 'uncached_tokens': 100, 'wall_s': seconds + 1}
                        for kind in ['warm', 'cold'] for turn in range(4)]
                (p / 'requests.jsonl').write_text('\n'.join(map(json.dumps, rows)))

    def test_interval_uses_two_boot_means(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d))
            row = report(Path(d), ['boot-a', 'boot-b'])['turns'][1]['warm']
            self.assertEqual(row['boot_mean_ttft_s'], [1, 3])
            self.assertEqual(row['mean_ttft_s'], 2)
            self.assertAlmostEqual(row['ci95_s'][1], 2 + 12.706204736)

    def test_paired_changes_keep_boot_units(self):
        row = paired([10, 20], [8, 16])
        self.assertEqual(row["absolute_change"], -3)
        self.assertAlmostEqual(row["percent_change"], -20)
        self.assertAlmostEqual(row["absolute_change_ci95"][1], -3 + 12.706204736)

    def test_guard_failure_rejects_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            (root / 'boot-a/rep-1.conditions.json').write_text(json.dumps({'passed': False}))
            with self.assertRaisesRegex(ValueError, 'guard failed'):
                report(root, ['boot-a', 'boot-b'])


if __name__ == '__main__':
    unittest.main()
