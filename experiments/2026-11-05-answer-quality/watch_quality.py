"""Out-of-band fail-fast observer; only the owned benchmark client may be stopped."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def regressions(native,candidate):
    assert native['complete']
    if 'phases' not in native:
        assert native['benchmark']==candidate['benchmark']=='RULER project protocol'
        for key in ('manifest_sha256','protocol','split','stage'):
            assert candidate.get(key)==native.get(key),key
        expected={r['id']:r for r in native['records']};seen=set();failures=[]
        for row in candidate['records']:
            assert row['id'] not in seen;seen.add(row['id']);old=expected[row['id']]
            assert all(row[k]==old[k] for k in ('prompt_sha256','task','family','generator_length'))
            complete=row['status']=='complete' and row.get('finish_reason')=='stop'
            assert 0<=row['score']<=100 and row['correct']==(complete and row['score']==100)
            baseline_complete=old['status']=='complete' and old.get('finish_reason')=='stop'
            if row['score']<old['score'] or (baseline_complete and not complete):
                failures.append({'id':row['id'],'native_score':old['score'],'candidate_score':row['score'],
                                 'candidate_status':row['status']})
        return failures
    for key in ('split','selection_sha256','dataset_sha256','protocol'):
        assert candidate[key]==native[key],key
    failures=[]
    phases={p['name']:p for p in native['phases']}
    for phase in candidate['phases']:
        old=phases[phase['name']]
        assert (phase['concurrency'],phase['cache_reuse'])==(old['concurrency'],old['cache_reuse'])
        expected={r['id']:r for r in old['records']};seen=set()
        for row in phase['records']:
            assert row['id'] not in seen;seen.add(row['id'])
            baseline=expected[row['id']]
            assert all(row[k]==baseline[k] for k in ('prompt_sha256','expected','domain'))
            complete=row['status']=='complete' and row.get('finish_reason')=='stop'
            assert row['correct']==(complete and row.get('predicted')==row['expected'])
            baseline_complete=baseline['status']=='complete' and baseline.get('finish_reason')=='stop'
            if (baseline['correct'] and not row['correct']) or (baseline_complete and not complete):
                failures.append({'phase':phase['name'],'id':row['id'],'native_correct':baseline['correct'],
                                 'candidate_correct':row['correct'],'candidate_status':row['status']})
    return failures


def client_pids(root,out,proc_root=Path('/proc')):
    """Find exact measure/evaluate parent-child pairs for this output path."""
    processes={}
    for path in proc_root.iterdir():
        if not path.name.isdecimal():continue
        try:
            argv=(path/'cmdline').read_bytes().split(b'\0');argv=[a.decode() for a in argv if a]
            if '--out' not in argv or argv[argv.index('--out')+1]!=str(out):continue
            parent=int((path/'stat').read_text().rsplit(')',1)[1].split()[1])
            processes[int(path.name)]=(parent,argv)
        except (OSError,UnicodeError,ValueError,IndexError):continue
    found=[]
    clients=[root/('experiments/'+folder+'/evaluate.py') for folder in ('2026-11-05-answer-quality','2026-11-05-context-quality')]
    parents=[root/('experiments/'+folder+'/measure.py') for folder in ('2026-11-05-answer-quality','2026-11-05-quality-matrix')]
    for pid,(parent,argv) in processes.items():
        if not any(str(p) in argv for p in clients):continue
        if parent not in processes:continue
        if not any(str(p) in processes[parent][1] for p in parents):continue
        found.append(pid)
    return sorted(found)


def main():
    p=argparse.ArgumentParser();p.add_argument('--window',type=Path,required=True)
    p.add_argument('--controller-root',type=Path,required=True);p.add_argument('--poll-seconds',type=float,default=5)
    args=p.parse_args();window=args.window.resolve();root=args.controller_root.resolve()
    assert window.is_relative_to(Path.home()/'spark-lab/results') and root.is_dir()
    assert args.poll_seconds>=1
    identity=json.loads((window/'window.json').read_text());token=identity['token']
    assert identity['holder'] in ('independent-answer-quality-window','lsa-independent-quality-window')
    allowed=os.sched_getaffinity(0)-set(range(5,10))-set(range(15,20))
    if allowed:os.sched_setaffinity(0,allowed)
    (window/'quality-observer.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'controller_root':str(root),'serving_commit':identity['commit'],'poll_seconds':args.poll_seconds,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'scope':'out-of-band public receipt checks; no server, activation or reference-model calls; stop only the owned failed benchmark client'},indent=2)+'\n')
    while not (window/'restored.json').exists():
        try:hold=json.loads((Path.home()/'spark-hold.json').read_text())
        except FileNotFoundError:return  # Restoration removes holds before its final receipt.
        assert hold['token']==token
        try:
            native_bytes=(window/'answers-native-a.json').read_bytes();native=json.loads(native_bytes)
            if not native.get('complete'):time.sleep(args.poll_seconds);continue
            for label in ('lsa-a','lsa-b'):
                path=window/('answers-'+label+'.json')
                if not path.exists():continue
                candidate_bytes=path.read_bytes();candidate=json.loads(candidate_bytes);failures=regressions(native,candidate)
                if not failures:continue
                assert json.loads((Path.home()/'spark-hold.json').read_text())['token']==token
                pids=client_pids(root,path)
                assert len(pids)<=1,'ambiguous benchmark client identity'
                (window/'quality-observer-native-snapshot.json').write_bytes(native_bytes)
                (window/'quality-observer-candidate-snapshot.json').write_bytes(candidate_bytes)
                receipt={'at':time.time(),'label':label,'failures':failures,'client_pids':pids,
                    'native_sha256':hashlib.sha256(native_bytes).hexdigest(),
                    'candidate_partial_sha256':hashlib.sha256(candidate_bytes).hexdigest(),
                    'action':'terminate exact owned evaluate child; existing coordinator records failure and restores every node',
                    'request_speed_qualified':False}
                target=window/'quality-reject-observer.json';assert not target.exists()
                target.write_text(json.dumps(receipt,indent=2)+'\n')
                for pid in pids:
                    if pid in client_pids(root,path):
                        try:os.kill(pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                print(json.dumps(receipt),flush=True);return
        except (FileNotFoundError,json.JSONDecodeError):
            pass  # A live receipt may be between its truncate and write.
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
