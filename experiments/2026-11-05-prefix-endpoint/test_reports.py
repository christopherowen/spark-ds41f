import json
import tempfile
import unittest
from pathlib import Path
from tables_arms import report
from compare import paired, comparison
import shutil


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

    def comparison_fixture(self, root):
        self.fixture(root)
        for source, label in [('boot-a', 'control-tp3-a'), ('boot-b', 'control-tp3-b')]:
            (root / source).rename(root / label)
            for p in (root / label).glob('rep-*'):
                if not p.is_dir():
                    continue
                summary = json.loads((p / 'summary.json').read_text())
                summary['distinct_agent_output_tokens_per_s'] = 10
                (p / 'summary.json').write_text(json.dumps(summary))
                rows = [json.loads(line) for line in (p / 'requests.jsonl').read_text().splitlines()]
                for row in rows:
                    row.update(prompt_sha256=row['label'], token_ids=[1, 2], logprobs=[-1, -2],
                               output_tokens_per_s=2 / row['wall_s'])
                (p / 'requests.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            shutil.copytree(root / label, root / label.replace('control', 'candidate'))
        return ['candidate-tp3-a', 'control-tp3-a', 'control-tp3-b', 'candidate-tp3-b']

    def test_full_comparison_keeps_zero_change_and_checks_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            labels = self.comparison_fixture(root)
            result = comparison(root, labels)
            self.assertTrue(result['passed_output_equality'])
            self.assertTrue(all(row['absolute_change'] == 0 for row in result['metrics']))

    def test_cross_build_output_mismatch_rejects_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            labels = self.comparison_fixture(root)
            p = root / 'candidate-tp3-a/rep-0/requests.jsonl'
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rows[0]['token_ids'] = [9, 2]
            p.write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, 'output mismatch'):
                comparison(root, labels)

    def test_guard_failure_rejects_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            (root / 'boot-a/rep-1.conditions.json').write_text(json.dumps({'passed': False}))
            with self.assertRaisesRegex(ValueError, 'guard failed'):
                report(root, ['boot-a', 'boot-b'])


if __name__ == '__main__':
    unittest.main()
