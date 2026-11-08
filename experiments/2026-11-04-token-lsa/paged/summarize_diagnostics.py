"""Report diagnostic request speed and fidelity separately, never hide failures."""
import argparse
import json
from pathlib import Path
import statistics
from summarize_requests import interval


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def compare(old,new):
    a,b = interval(old),interval(new)
    return {'native_two_boot_values':old,'candidate_two_boot_values':new,
            'native_over_candidate':statistics.mean(old)/statistics.mean(new),
            'joint95_ratio_interval':[a[0]/b[1],a[1]/b[0]] if min(a+b) > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('window',type=Path)
    out = parser.parse_args().window
    jobs = read(out/'jobs.json')
    assert jobs and jobs['diagnostic_timing']
    groups = {}
    for label,profile in jobs['jobs']:
        groups.setdefault(profile,[]).append(label)
    assert all(len(labels) == 2 for labels in groups.values()) and 'control' in groups
    native = groups.pop('control')
    results = {label:read(out/('requests-'+label+'.json')) for label,_ in jobs['jobs']}
    complete = all(r and len(r['cases']) == 3 for r in results.values())
    report = {'complete':complete,'scope':'synthetic request timing, diagnostic until every fidelity gate passes',
              'uncertainty':'two fresh boots per arm; df1 Student-t component97.5% and Bonferroni joint95% ratio intervals; normality unverified',
              'restored':read(out/'restored.json'),
              'native_repeat_gates':{s:read(out/(s+'.txt')) for s in ('native-baseline-repeat','native-parallel-repeat')},
              'presets':{}}
    for profile,labels in groups.items():
        gates = {label:{str(c):read(out/('token-comparison-'+label+('-c4' if c == 4 else '')+'.txt')) for c in (1,4)} for label in labels+native if label != 'control-a'}
        entry = {'raw_token_gates':gates,'comparisons':[],'qualified_comparison':False}
        report['presets'][profile] = entry
        if not complete:
            continue
        timed_matches = []
        for case_index in range(3):
            arms = {label:results[label]['cases'][case_index] for label in native+labels}
            geometry = {(c['length'],c['concurrency']) for c in arms.values()}
            assert len(geometry) == 1
            length,concurrency = next(iter(geometry))
            for phase in ('cold','cached'):
                rows,batches = {},{}
                for label,c in arms.items():
                    rows[label] = c['cold'] if phase == 'cold' else [r for group in c['cached_repeats'] for r in group]
                    batches[label] = [c['cold_batch']] if phase == 'cold' else c['cached_batches']
                    assert len(rows[label]) == concurrency*(1 if phase == 'cold' else 3)
                    assert all(r['usage']['prompt_tokens'] == length for r in rows[label])
                assert len({r['input_sha256'] for row in rows.values() for r in row}) == 1
                matched = len({(r['usage']['completion_tokens'],r['choices_sha256']) for row in rows.values() for r in row}) == 1
                timed_matches.append(matched)
                latency = {label:statistics.median(r['seconds'] for r in row) for label,row in rows.items()}
                tps = {label:statistics.median(b['aggregate_output_tokens_per_s'] for b in row) for label,row in batches.items()}
                c = {'length':length,'concurrency':concurrency,'phase':phase,'timed_response_digest_count_match':matched,
                     'request_seconds':compare([latency[x] for x in native],[latency[x] for x in labels]),
                     'aggregate_output_tokens_per_s':{label:tps[label] for label in native+labels},
                     'throughput_candidate_over_native':statistics.mean(tps[x] for x in labels)/statistics.mean(tps[x] for x in native),
                     'actual_cached_tokens':{label:[r['usage'].get('prompt_tokens_details',{}).get('cached_tokens') for r in row] for label,row in rows.items()},
                     'completion_token_counts':{label:[r['usage']['completion_tokens'] for r in row] for label,row in rows.items()}}
                a,b = interval([tps[x] for x in native]),interval([tps[x] for x in labels])
                c['throughput_joint95_ratio_interval'] = [b[0]/a[1],b[1]/a[0]] if min(a+b) > 0 else None
                entry['comparisons'].append(c)
        conditions = {label:read(out/('requests-'+label+'-conditions.json')) for label in native+labels}
        entry['conditions_passed'] = all(v and not v['halted'] and not v['fan_stop_failures']
            and len(v['thermal']) == 3 and not any(t.get('thermal_slowdown_ms') for t in v['thermal'].values()) for v in conditions.values())
        entry['fidelity_passed'] = bool(all(v and v['passed'] for group in gates.values() for v in group.values())
            and all(v and v['passed'] for v in report['native_repeat_gates'].values()) and all(timed_matches))
        entry['qualified_comparison'] = entry['fidelity_passed'] and entry['conditions_passed'] and bool(report['restored'] and report['restored'].get('restored') and report['restored'].get('holds_removed'))
        entry['verdict'] = 'fidelity-qualified comparison within declared finite screen; owner promotion still required' if entry['qualified_comparison'] else 'diagnostic only; does not establish an accepted speedup'
    if not complete:
        report['missing_or_incomplete'] = [name for name,r in results.items() if not r or len(r['cases']) != 3]
    (out/'diagnostic-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
