"""Matched process medians; never turn query-only timings into request claims."""
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def interval(values):
    mean = statistics.mean(values)
    # df1,97.5% two-sided components, giving a Bonferroni joint95% ratio.
    delta = math.tan(math.pi*(.9875-.5))*abs(values[0]-values[1])/2
    return [mean-delta,mean+delta]


def main():
    groups = []
    for suffix,kind in [('', 'captured'),('-long','replicated512K'),('-batch','captured-repeated8queries'),('-long-batch','replicated512K-repeated8queries')]:
        folders = {arm:[HERE/'runs'/f'shortlist-{arm}{suffix}-{i:02d}' for i in (1,2)] for arm in ('control','candidate')}
        if not all((p/'result.json').exists() for paths in folders.values() for p in paths):
            continue
        runs = {}
        for arm,paths in folders.items():
            runs[arm] = []
            for path in paths:
                verdict,result = json.loads((path/'verdict.json').read_text()),json.loads((path/'result.json').read_text())
                assert verdict['passed'] and verdict['exit'] == 0 and result['passed']
                assert verdict['image'] == 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'
                assert result['mode'] == arm and len(result['cases']) == 4
                runs[arm].append(result['cases'])
        for index in range(4):
            cases = {arm:[run[index] for run in runs[arm]] for arm in runs}
            signature = lambda x:(x['snapshot'],x['extent'],x['rows'],x['page_size'],x['planned_width'],x['source'],x['synthetic_repeated_pages'])
            assert len({signature(x) for row in cases.values() for x in row}) == 1
            values = {arm:[x['query_ms_median'] for x in row] for arm,row in cases.items()}
            c,s = interval(values['control']),interval(values['candidate'])
            bounds = [c[0]/s[1],c[1]/s[0]] if min(c+s) > 0 else None
            construction = [x['construction']['construction_wall_ms'] for x in cases['candidate']]
            saving = statistics.mean(values['control'])-statistics.mean(values['candidate'])
            groups.append({'workload':kind,'snapshot':cases['control'][0]['snapshot'],
                'rows':cases['control'][0]['rows'],'extent':cases['control'][0]['extent'],
                'control_query_ms_two_processes':values['control'],'candidate_query_ms_two_processes':values['candidate'],
                'query_speedup':statistics.mean(values['control'])/statistics.mean(values['candidate']),
                'descriptive_joint95_ratio_interval':bounds,'construction_ms_two_processes':construction,
                'construction_break_even_queries_at_mean_saving':statistics.mean(construction)/saving if saving > 0 else None})
    result = {'scope':'snapshot query-only timings; replicated cases are synthetic; no full-model token/request qualification',
              'uncertainty':'two process medians, df1; normality unverified; retain raw samples and both cold/warm construction costs',
              'performance_gate':'full-request cost must include essential construction and updates',
              'generated_token_gate':'pending; snapshot tests cannot establish it','points':groups}
    assert groups
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Shortlist kernel screen','',result['scope']+'.','',
        '| Workload | Snapshot | Control ms (two processes) | Shortlist ms (two processes) | Query speedup |',
        '| --- | --- | --- | --- | ---: |']
    for row in groups:
        a,b=row['control_query_ms_two_processes'],row['candidate_query_ms_two_processes']
        lines.append(f"| {row['workload']} | {row['snapshot']} | {a[0]:.4f}/{a[1]:.4f} | {b[0]:.4f}/{b[1]:.4f} | {row['query_speedup']:.2f}x |")
    lines += ['', 'Full confidence intervals, cold/warm construction costs, break-even accounting',
              'and all underlying samples are retained in `results.json` and `runs/`.',
              'Construction is required work outside these query timers, not a free input.',
              'Small captured single-query cases lose to native search. Long replicated',
              'cases establish a query-kernel scaling result, not real long-request speed.',
              'Generated-token equivalence and serving cache maintenance remain pending.']
    (HERE/'results.md').write_text('\n'.join(lines)+'\n')
    print('matched process receipts and explicit construction accounting summarized')


if __name__ == '__main__':
    main()
