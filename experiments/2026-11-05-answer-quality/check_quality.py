"""Fail closed on any observed answer regression; LongBench receipts only."""
import argparse
import hashlib
import json
from pathlib import Path


def check(native, candidate):
    if len(native) != 2 or len(candidate) != 2:
        raise ValueError('two independent boot receipts are required per arm')
    runs = dict(zip(('native-a','native-b','lsa-a','lsa-b'), native+candidate))
    first = native[0]
    if not first.get('phases'):
        raise ValueError('no benchmark phases')
    for name, run in runs.items():
        if not run.get('complete'):
            raise ValueError(f'{name}: unfinished workload')
        for key in ('split','selection_sha256','dataset_sha256','protocol'):
            if key not in first or run.get(key) != first[key]:
                raise ValueError(f'{name}: mismatched {key}')
        if len(run['phases']) != len(first['phases']):
            raise ValueError(f'{name}: mismatched phase coverage')
    failures, phases, documents = [], [], set()
    for index, baseline in enumerate(first['phases']):
        current = {name:run['phases'][index] for name,run in runs.items()}
        identity = [(r['id'],r['prompt_sha256'],r['expected'],r['domain']) for r in baseline['records']]
        if not identity or len({r[0] for r in identity}) != len(identity):
            raise ValueError('empty or duplicated task identities')
        documents.update(r[0] for r in identity)
        for name, phase in current.items():
            for key in ('name','concurrency','cache_reuse'):
                if phase[key] != baseline[key]:
                    raise ValueError(f'{name}: mismatched phase {key}')
            if [(r['id'],r['prompt_sha256'],r['expected'],r['domain']) for r in phase['records']] != identity:
                raise ValueError(f'{name}: mismatched task identities or coverage')
            if phase['total'] != len(identity):
                raise ValueError(f'{name}: inconsistent task count')
            for row in phase['records']:
                if type(row['correct']) is not bool or row['expected'] not in 'ABCD' or len(row['expected']) != 1:
                    raise ValueError(f'{name}: invalid independent answer receipt')
                graded = (row['status'] == 'complete' and row.get('finish_reason') == 'stop'
                          and row.get('predicted') == row['expected'])
                if graded != row['correct']:
                    raise ValueError(f'{name}/{row["id"]}: inconsistent grade')
        wins, losses, incomplete, new_incomplete, unstable = [], [], [], [], []
        strata = {}
        for pos, (case, _, _, domain) in enumerate(identity):
            rows = {name:p['records'][pos] for name,p in current.items()}
            for name,row in rows.items():
                if row['status'] != 'complete' or row.get('finish_reason') != 'stop':
                    item={'id':case,'boot':name,'status':row['status']}
                    incomplete.append(item)
                    if name.startswith('lsa-') and any(rows[n]['status']=='complete' and rows[n].get('finish_reason')=='stop' for n in ('native-a','native-b')):
                        new_incomplete.append(item)
            for arm in ('native','lsa'):
                a,b = rows[arm+'-a'],rows[arm+'-b']
                if (a['correct'],a.get('predicted')) != (b['correct'],b.get('predicted')):
                    unstable.append({'id':case,'arm':arm})
            old = any(rows[n]['correct'] for n in ('native-a','native-b'))
            new = all(rows[n]['correct'] for n in ('lsa-a','lsa-b'))
            if old and not new:losses.append(case)
            if not old and new:wins.append(case)
            bucket = strata.setdefault(domain,{'documents':0,'wins':[],'losses':[]})
            bucket['documents'] += 1
            if case in wins:bucket['wins'].append(case)
            if case in losses:bucket['losses'].append(case)
        row = {'name':baseline['name'],'independent_documents':len(identity),
               'wins':wins,'losses':losses,'incomplete':incomplete,'new_incomplete':new_incomplete,'unstable':unstable,'domains':strata}
        phases.append(row)
        if losses:failures.append({'phase':row['name'],'reason':'new incorrect or incomplete answers','ids':losses})
        if new_incomplete:
            failures.append({'phase':row['name'],'reason':'new candidate incompletion relative to a completed native response'})
        if unstable:failures.append({'phase':row['name'],'reason':'answer decisions differ between repeated boots'})
    return {'passed':not failures,'failures':failures,'phases':phases,
            'gate_version':'observed-answer-and-completion-no-loss-v2',
            'independent_documents':len(documents),'promotion_qualified':False,
            'scope':'observed final-answer no-loss gate only; no claim about untested inputs or speed',
            'thinking_compared':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native',type=Path,nargs=2,required=True)
    parser.add_argument('--candidate',type=Path,nargs=2,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():raise ValueError('refuse to overwrite a receipt')
    paths = args.native+args.candidate
    try:
        for arm in (args.native,args.candidate):
            if hashlib.sha256(arm[0].read_bytes()).digest() == hashlib.sha256(arm[1].read_bytes()).digest():
                raise ValueError('duplicate receipts do not establish independent boots')
        result = check([json.loads(p.read_text()) for p in args.native],
                       [json.loads(p.read_text()) for p in args.candidate])
    except (KeyError,TypeError,ValueError) as error:
        result = {'passed':False,'promotion_qualified':False,'error':str(error)}
    result['input_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':raise SystemExit(main())
