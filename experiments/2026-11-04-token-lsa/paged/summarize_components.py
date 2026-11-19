"""Two-process component receipts, including broad/undefined uncertainty."""
import argparse
import json
from pathlib import Path
import statistics

from summarize_requests import interval


def ratio(old,new):
    a,b = interval(old),interval(new)
    return {'old_two_process_medians_ms':old,'new_two_process_medians_ms':new,
            'speedup':statistics.mean(old)/statistics.mean(new),
            'joint95_speedup_interval':[a[0]/b[1],a[1]/b[0]] if min(a+b) > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('window',type=Path)
    parser.add_argument('--kind',choices=['compact','pipeline'],required=True)
    args = parser.parse_args()
    if args.kind == 'compact':
        names,metric = ('scan','direct'),'median_ms'
    else:
        names,metric = ('row8','row256'),'query_graph_gpu_ms_median'
    runs = {}
    for name in names:
        runs[name] = []
        for i in (1,2):
            path = args.window/f'{args.kind}-{name}-{i:02d}'
            verdict = json.loads((path/'verdict.json').read_text())
            result = json.loads((path/'result.json').read_text())
            assert verdict['passed'] and verdict['exit'] == 0 and result['passed']
            runs[name].append((result,verdict))
    assert len({v['node'] for group in runs.values() for _,v in group}) == 1
    assert len({v['image'] for group in runs.values() for _,v in group}) == 1
    count = len(runs[names[0]][0][0]['cases'])
    result = {'scope':'synthetic component only; no request speedup or model-fidelity claim',
              'uncertainty':'two process medians; df1 Student-t component97.5% intervals, Bonferroni joint95% ratio; normality unverified',
              'cases':[]}
    for i in range(count):
        cases = {name:[run['cases'][i] for run,_ in group] for name,group in runs.items()}
        keys = ('width','extent','page','budget','rows')
        assert len({tuple(c[k] for k in keys) for row in cases.values() for c in row}) == 1
        if args.kind == 'compact':
            assert len({c['input_sha256'] for row in cases.values() for c in row}) == 1
        else:
            assert len({c['snapshot_sha256'] for row in cases.values() for c in row}) == 1
        c = {k:cases[names[0]][0][k] for k in keys}
        c['graph_gpu'] = ratio(*[[case[metric] for case in cases[name]] for name in names])
        if args.kind == 'pipeline':
            c['eager_wall'] = ratio(*[[case['query_eager_wall_ms_median'] for case in cases[name]] for name in names])
            c['construction_wall_ms'] = {name:[case['construction_wall_ms'] for case in row] for name,row in cases.items()}
            c['workspace_bytes'] = {name:[case['workspace_bytes'] for case in row] for name,row in cases.items()}
        result['cases'].append(c)
    (args.window/(args.kind+'-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
