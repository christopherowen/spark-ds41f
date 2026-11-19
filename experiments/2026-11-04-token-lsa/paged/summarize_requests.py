"""Compare paired boot receipts without hiding cold work or token failures."""
import argparse
import json
import math
from pathlib import Path
import statistics


def interval(values):
    assert len(values) == 2
    center = statistics.mean(values)
    delta = math.tan(math.pi*(.9875-.5))*abs(values[0]-values[1])/2
    return [center-delta,center+delta]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('window',type=Path)
    options = parser.parse_args()
    out = options.window
    names = ['control-a','shortlist-a','shortlist-b','control-b']
    files = [out/('requests-'+name+'.json') for name in names]
    result = {'complete':all(p.exists() and len(json.loads(p.read_text()).get('cases',[])) == 3 for p in files),'comparisons':[],
              'scope':'synthetic C36 cold/cached request latency; no production-agent throughput claim',
              'uncertainty':'two boot medians per arm; df1 Student-t component97.5% intervals and Bonferroni joint95% ratio; normality unverified',
              'token_scope':'separate serial and four-concurrent raw-token gates; timed requests do not capture raw IDs'}
    if result['complete']:
        runs = {name:json.loads(p.read_text()) for name,p in zip(names,files)}
        for case in range(3):
            geometry = {(r['cases'][case]['length'],r['cases'][case]['concurrency']) for r in runs.values()}
            assert len(geometry) == 1
            length,concurrency = next(iter(geometry))
            for phase in ('cold','cached_repeats'):
                rows = {}
                for name,run in runs.items():
                    current = run['cases'][case][phase]
                    rows[name] = current if phase == 'cold' else [r for repeat in current for r in repeat]
                    assert len(rows[name]) == concurrency*(1 if phase == 'cold' else 3)
                    assert all(r['usage']['prompt_tokens'] == length for r in rows[name])
                signatures = {(r['input_sha256'],r['usage']['completion_tokens'],r['choices_sha256'])
                              for row in rows.values() for r in row}
                matched = len(signatures) == 1
                values = {arm:[statistics.median(r['seconds'] for r in rows[name]) for name in labels]
                          for arm,labels in [('control',['control-a','control-b']),('candidate',['shortlist-a','shortlist-b'])]}
                control,candidate = interval(values['control']),interval(values['candidate'])
                ci = [control[0]/candidate[1],control[1]/candidate[0]] if min(control+candidate) > 0 else None
                cached = {name:[r['usage'].get('prompt_tokens_details',{}).get('cached_tokens') for r in row]
                          for name,row in rows.items()}
                result['comparisons'].append({'length':length,'concurrency':concurrency,'phase':phase,
                    'input_output_digest_count_match':matched,'seconds_two_boot_medians':values,
                    'speedup':statistics.mean(values['control'])/statistics.mean(values['candidate']),
                    'speedup_joint95_interval':ci,'actual_cached_token_counts':cached})
        comparisons = [out/('token-comparison-'+name+'.txt') for name in names if name != 'control-a']
        result['raw_token_gates'] = {p.stem:json.loads(p.read_text()) if p.exists() else None for p in comparisons}
        result['all_serial_token_gates_pass'] = all(v and v['passed'] for v in result['raw_token_gates'].values())
        result['parallel_token_gates'] = {
            name:json.loads((out/('token-comparison-'+name+'-c4.txt')).read_text())
            if (out/('token-comparison-'+name+'-c4.txt')).exists() else None
            for name in names if name != 'control-a'}
        result['all_parallel_token_gates_pass'] = all(v and v['passed'] for v in result['parallel_token_gates'].values())
        result['native_stability'] = {
            name:json.loads((out/(name+'.txt')).read_text()) if (out/(name+'.txt')).exists() else None
            for name in ['native-baseline-repeat','native-parallel-repeat']}
        result['all_timed_response_checks_pass'] = all(x['input_output_digest_count_match'] for x in result['comparisons'])
        result['comparison_fidelity_passed'] = bool(result['all_serial_token_gates_pass']
            and result['all_parallel_token_gates_pass'] and result['all_timed_response_checks_pass']
            and all(v and v['passed'] for v in result['native_stability'].values()))
    else:
        result['missing'] = [p.name for p in files if not p.exists()]
        result['conclusion'] = 'Incomplete paired boots; no request speedup established.'
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
