#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize paired arms with the fresh process as the independent unit.

Two-process 95% Student intervals use df=1 (12.706), not 62 independent
trials. These intervals can be wide; crossing zero is inconclusive.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as stats


def interval_two(values):
    assert len(values) == 2
    mean = stats.mean(values)
    half = 12.706204736 * stats.stdev(values) / math.sqrt(2)
    return [mean - half, mean + half]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipts', nargs=2, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in args.receipts]
    assert {r['boot'] for r in records} == {1, 2}
    assert all(r['status'] == 'passed' for r in records)
    assert records[0]['script_sha256'] == records[1]['script_sha256']
    assert records[0]['source_hashes'] == records[1]['source_hashes']
    assert all(r['restored_limit'] == r['original_limit'] for r in records)
    keyed = [{(c['tp'], c['rows'], c['expert_stream_stress']): c
              for c in r['cases']} for r in records]
    assert keyed[0].keys() == keyed[1].keys()
    result = {'method': 'two independent process means; paired arm differences; '
                       'two-sided 95% Student interval, df=1; negative delta is faster',
              'scope': 'native packed Markov component, not serving or quality',
              'receipts': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in args.receipts], 'cases': []}
    for key in sorted(keyed[0]):
        cases = [k[key] for k in keyed]
        assert all(c['all_bits_equal'] for c in cases)
        assert cases[0]['weight_sha256'] == cases[1]['weight_sha256']
        arms = sorted({s['arm'] for s in cases[0]['samples']})
        comparisons = [('driver_default', arm) for arm in arms if arm != 'driver_default']
        comparisons += [('prefetch_default', arm) for arm in
                        ('window_default', 'both_default', 'window_small', 'both_small', 'window_fit', 'both_fit')]
        comparisons += [('partition_small', 'window_small'), ('partition_fit', 'window_fit'), ('demand', 'null')]
        case_result = {'tp': key[0], 'rows': key[1], 'expert_stream_stress': key[2],
                       'payload_bytes': cases[0]['payload_bytes'], 'comparisons': []}
        for before, after in comparisons:
            for metric in ('us', 'after_us'):
                boot_values = []
                for c in cases:
                    a = {s['trial']: s[metric] for s in c['samples'] if s['arm'] == before}
                    b = {s['trial']: s[metric] for s in c['samples'] if s['arm'] == after}
                    assert a.keys() == b.keys() and len(a) >= 31
                    av, bv = stats.mean(a.values()), stats.mean(b.values())
                    boot_values.append({'before': av, 'after': bv,
                                        'delta': bv - av, 'percent': 100 * (bv / av - 1),
                                        'paired_trials': len(a)})
                case_result['comparisons'].append({
                    'before_arm': before, 'after_arm': after, 'metric': metric,
                    'before_us': stats.mean(v['before'] for v in boot_values),
                    'after_us': stats.mean(v['after'] for v in boot_values),
                    'delta_us': stats.mean(v['delta'] for v in boot_values),
                    'percent': stats.mean(v['percent'] for v in boot_values),
                    'delta_95_us': interval_two([v['delta'] for v in boot_values]),
                    'percent_95': interval_two([v['percent'] for v in boot_values]),
                    'boots': boot_values,
                })
        result['cases'].append(case_result)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
