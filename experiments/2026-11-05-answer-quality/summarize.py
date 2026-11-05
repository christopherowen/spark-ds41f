"""Independent final-answer accuracy and useful throughput, without token identity."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics


def interval(values):
    assert len(values)==2
    center=statistics.mean(values)
    delta=math.tan(math.pi*(.9875-.5))*abs(values[0]-values[1])/2
    return [center-delta,center+delta]


def speed(old,new):
    a,b=interval(old),interval(new)
    return {'native_two_boot_values':old,'lsa_two_boot_values':new,
            'native_over_lsa':statistics.mean(old)/statistics.mean(new),
            'joint95_ratio_interval':[a[0]/b[1],a[1]/b[0]] if min(a+b)>0 else None}


def paired(native,lsa):
    assert [(r['id'],r['prompt_sha256'],r['expected']) for r in native] == [(r['id'],r['prompt_sha256'],r['expected']) for r in lsa]
    wins=[a['id'] for a,b in zip(native,lsa) if not a['correct'] and b['correct']]
    losses=[a['id'] for a,b in zip(native,lsa) if a['correct'] and not b['correct']]
    d=len(wins)+len(losses)
    p=sum(math.comb(d,k) for k in range(len(wins),d+1))/2**d if d else 1.
    return {'n_independent_documents':len(native),'wins':wins,'losses':losses,
            'accuracy_delta':(len(wins)-len(losses))/len(native),
            'exact_one_sided_discordant_pair_p':p,
            'scope':'paired binary task scores; assumes independent representative documents; pilot is not confirmatory'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('window',type=Path);out=parser.parse_args().window
    jobs=json.loads((out/'jobs.json').read_text())
    runs={label:json.loads((out/('answers-'+label+'.json')).read_text()) if (out/('answers-'+label+'.json')).exists() else None for label,_ in jobs['jobs']}
    complete=all(r and r.get('complete') for r in runs.values())
    result={'complete':complete,'split':jobs['split'],'promotion_qualified':False,
            'rule':'grade completed final answers against independent ground truth; reasoning may differ; all runtime counts',
            'uncertainty':'two boot performance intervals assume normal boot means (unverified), df1 component97.5%/joint95%; quality sample size is unique documents, never duplicated boots',
            'restored':json.loads((out/'restored.json').read_text()) if (out/'restored.json').exists() else None,'phases':[]}
    if complete:
        assert len({r['selection_sha256'] for r in runs.values()})==1
        assert len({json.dumps(r['protocol'],sort_keys=True) for r in runs.values()})==1
        for index in range(len(runs['native-a']['phases'])):
            phases={name:r['phases'][index] for name,r in runs.items()}
            assert len({(p['name'],p['concurrency'],p['cache_reuse']) for p in phases.values()})==1
            row={'name':phases['native-a']['name'],'per_boot':{},'paired_quality':{},'strata':{}}
            for name,p in phases.items():
                row['per_boot'][name]={k:p[k] for k in ['wall_seconds','correct','total','accuracy','correct_answers_per_second','median_request_seconds']}
                row['per_boot'][name]['statuses']={s:sum(r['status']==s for r in p['records']) for s in sorted({r['status'] for r in p['records']})}
            for a,b in itertools.product(['native-a','native-b'],['lsa-a','lsa-b']):
                row['paired_quality'][a+'/'+b]=paired(phases[a]['records'],phases[b]['records'])
            keys={r['domain'] for p in phases.values() for r in p['records']}
            for domain in sorted(keys):
                row['strata'][domain]={name:{'correct':sum(r['correct'] for r in p['records'] if r['domain']==domain),
                    'total':sum(r['domain']==domain for r in p['records'])} for name,p in phases.items()}
            row['wall_time']=speed([phases[n]['wall_seconds'] for n in ['native-a','native-b']],
                                   [phases[n]['wall_seconds'] for n in ['lsa-a','lsa-b']])
            old=[phases[n]['correct_answers_per_second'] for n in ['native-a','native-b']]
            new=[phases[n]['correct_answers_per_second'] for n in ['lsa-a','lsa-b']]
            a,b=interval(old),interval(new)
            row['useful_throughput']={'native_two_boot_values':old,'lsa_two_boot_values':new,
                'lsa_over_native':statistics.mean(new)/statistics.mean(old) if statistics.mean(old)>0 else None,
                'joint95_ratio_interval':[b[0]/a[1],b[1]/a[0]] if min(a+b)>0 else None}
            row['all_boot_pairings_favor_lsa_quality']=all(q['accuracy_delta']>0 and q['exact_one_sided_discordant_pair_p']<.05 for q in row['paired_quality'].values())
            result['phases'].append(row)
        conditions={name:json.loads((out/('answers-'+name+'-conditions.json')).read_text())
                    if (out/('answers-'+name+'-conditions.json')).exists() else None for name in runs}
        result['conditions_passed']=all(v and not v['halted'] and not v['fan_stop_failures']
            and len(v['thermal'])==len(jobs.get('nodes',['sp5','sp6','sp7']))
            and not any(t.get('thermal_slowdown_ms') for t in v['thermal'].values()) for v in conditions.values())
        result['conclusion']='Pilot evidence only; held-out completed-answer and performance confirmation required.' if jobs['split']=='pilot' else 'Inspect paired quality, useful speed and every task/context category before any superiority or promotion decision.'
    else:
        result['missing_or_incomplete']=[name for name,r in runs.items() if not r or not r.get('complete')]
        result['conclusion']='Incomplete benchmark window; no completed-answer quality or performance qualification.'
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
