"""Owner-authorized, coordinated page-shortlist qualification on sp5-sp7.

Native token-scoped leases, watchdog and the tracked C37 restoration recipe are
shared with the earlier request window. No one-rank serving launch is permitted.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = HERE.parent.parent/'2026-11-04-clustered-indexer/serving/run_window.py'
spec = importlib.util.spec_from_file_location('coordinated_request_window',source)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
w.REL = str(HERE.relative_to(ROOT))


def restore(token,out,stem,finish_heartbeat=None):
    if not w.owns(token):
        raise RuntimeError('reservation changed; refusing to overwrite another holder')
    code = w.run(w.spark('restore','doctor','--live'),out,stem+'-already-live',allow_failure=True)
    if code:
        w.start('restore',out,stem)
    check = w.each('''import json,subprocess
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True))[0]
roots=['/home/chris/projects/spark-ds41f','/home/chris/spark-lab/checkouts/clustered-serving']
print(json.dumps({'image':x['Image'],'running':x['State']['Running'],
 'checkouts':{p:{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
                 'status':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True)} for p in roots},
 'experiment_mounts':[m['Destination'] for m in x['Mounts'] if any(s in m['Source'] for s in ['clustered-indexer','token-lsa','selector-fitting','block-summary'])],
 'search_env':[e for e in x['Config']['Env'] if e.startswith(('VLLM_DS41_CLUSTERED_SEARCH=','VLLM_DS41_PAGE_'))]}))
''')
    assert all(v['image'] == w.IMAGE and v['running'] and not v['experiment_mounts'] and not v['search_env']
               and all(not c['status'] for c in v['checkouts'].values()) for v in check.values()),check
    (out/(stem+'-verification.json')).write_text(json.dumps(check,indent=2)+'\n')
    if finish_heartbeat is not None:
        finish_heartbeat()
    w.each(w.HOLD_CODE,{'action':'remove','token':token})
    (out/'restored.json').write_text(json.dumps({'at':w.stamp(),'restored':True,'holds_removed':True},indent=2)+'\n')


w.restore = restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--watchdog')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--diagnostic-profiles',nargs='+',help='label:profile arms; retain timings after explicit token mismatches')
    parser.add_argument('--candidate-profile',default='candidate',help='paired default candidate recipe')
    parser.add_argument('--kernel-jobs',nargs='+',help='isolated bundles before serving boots')
    parser.add_argument('--quality-profiles',nargs='+',help='predeclared label:profile token-only calibration')
    parser.add_argument('--reference-private',help='existing native capture/corpus directory; never copied to Git')
    options = parser.parse_args()
    out = Path(options.out).resolve()
    if socket.gethostname() != 'spark-0013':
        raise SystemExit('only the idle Spaceport sp5-sp7 group is authorized')
    if options.watchdog:
        return w.watchdog(options.watchdog,out)
    assert not (options.quality_profiles and options.diagnostic_profiles), 'choose calibration or diagnostic timing'
    out.mkdir(parents=True,exist_ok=False)
    if not options.quality_profiles and not options.diagnostic_profiles:
        assert all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in options.candidate_profile)
        assert (HERE/(options.candidate_profile+'.json')).is_file()
    if options.reference_private:
        assert (Path(options.reference_private)/'control-a.json').is_file()

    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','branch','-r','--contains',commit],cwd=ROOT,text=True).strip()
    holds = w.each(w.HOLD_CODE,{'action':'read'})
    assert not any(any(v.values()) for v in holds.values()),holds
    identity = w.each('''import json,subprocess
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True))[0]
roots=['/home/chris/projects/spark-ds41f','/home/chris/spark-lab/checkouts/clustered-serving']
print(json.dumps({'image':x['Image'],'running':x['State']['Running'],
 'containers':subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines(),
 'checkouts':{p:{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
                 'status':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True)} for p in roots}}))
''')
    assert all(v['image'] == w.IMAGE and v['running'] and v['containers'] == ['spark-ds41f']
               and all(not c['status'] for c in v['checkouts'].values()) for v in identity.values()),identity
    assert len({v['checkouts']['/home/chris/spark-lab/checkouts/clustered-serving']['head']
                for v in identity.values()}) == 1
    (out/'preflight-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    # Verify the exact currently restored recipe before any mutation.
    w.run(['bin/spark','--cluster-config','experiments/2026-11-04-clustered-indexer/serving/restore.json',
           'doctor','--live'],out,'preflight-doctor')
    command = w.spark('control','cluster','start')
    w.run(command,out,'runtime-plan')
    plan = (out/'runtime-plan.txt').read_text()
    review = {name:next((x for x in plan.splitlines() if x.startswith(name+':')),None)
              for name in ('live','shape')}
    assert all(review.values()) and 'unchanged' in review['shape'] and '--apply would refuse:' not in plan,review
    (out/'runtime-review.json').write_text(json.dumps(review,indent=2)+'\n')
    w.run(w.spark('control','cluster','sync','--revision',commit),out,'sync-plan')
    for job in options.kernel_jobs or []:
        bundle,label = job.split(':',1)
        assert bundle.startswith('sp8-') and Path(bundle).name == bundle and Path(label).name == label
        config = HERE.parent/bundle/'candidate.json'
        assert config.is_file()
        candidate = json.loads(config.read_text())
        assert candidate['image'] == 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'
        for relative,_ in candidate['mounts']:
            path = Path(relative)
            assert path.is_absolute() or (config.parent/path).is_file(),relative

    if options.preflight_only:
        return
    baseline = None
    for i in range(7):
        metrics = urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=15).read().decode()
        lines = [line for line in metrics.splitlines() if not line.startswith('#') and any(k in line for k in
                 ('num_requests_running{','num_requests_waiting{','request_success_total{'))]
        for line in lines:
            if 'num_requests_' in line:
                assert float(line.split()[-1]) == 0,line
        successes = [line for line in lines if 'request_success_total{' in line]
        if baseline is None:
            baseline = successes
        assert baseline == successes,'serving traffic changed during idle guard'
        (out/f'idle-{i}.json').write_text(json.dumps({'at':w.stamp(),'metrics':lines},indent=2)+'\n')
        if i < 6:
            time.sleep(5)
    if options.quality_profiles and not options.reference_private:
        parser.error('--reference-private is required with --quality-profiles')
    private = (Path(options.reference_private).resolve() if options.quality_profiles else
               Path.home()/'spark-lab/private/token-lsa-paged'/out.name)
    if options.quality_profiles:
        assert options.reference_private and (private/'control-a.json').is_file()
        assert (private/'requests.jsonl').is_file() and (private/'identity.json').is_file()
    for arm in options.quality_profiles or options.diagnostic_profiles or []:
        label,profile = arm.split(':',1)
        assert label and profile and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label+profile)
        assert (HERE/(profile+'.json')).is_file()
    if options.diagnostic_profiles:
        labels = [arm.split(':',1)[0] for arm in options.diagnostic_profiles]
        assert labels[0] == 'control-a' and len(set(labels)) == len(labels)
    token = uuid.uuid4().hex
    hold = {'holder':'page-shortlist-request-window','token':token,'since':w.stamp(),
            'heartbeat':w.stamp(),'expected_end':None,'nodes':['sp5','sp6','sp7'],'commit':commit,
            'rule':'exclusive token qualification and request timing; restore tracked C37 once',
            'production_config':w.REL+'/restore.json','controller_output':str(out)}
    acquired = []
    try:
        for node in w.NODES:
            w.remote(node,w.HOLD_CODE,{'action':'open','hold':hold})
            acquired.append(node)
    except Exception:
        for node in acquired:
            w.remote(node,w.HOLD_CODE,{'action':'remove','token':token})
        raise
    (out/'window.json').write_text(json.dumps(hold,indent=2)+'\n')
    stop,errors = threading.Event(),[]
    def beat():
        while not stop.wait(30):
            try:
                w.each(w.HOLD_CODE,{'action':'refresh','token':token,'heartbeat':w.stamp()})
            except Exception as error:
                errors.append(str(error))
                return
    thread = threading.Thread(target=beat,daemon=True)
    thread.start()
    def finish():
        stop.set()
        thread.join(timeout=35)
        assert not thread.is_alive(),'heartbeat writer did not stop; retaining leases'
    with (out/'watchdog.txt').open('w') as handle:
        subprocess.Popen([sys.executable,__file__,'--out',str(out),'--watchdog',token],cwd=ROOT,
                         stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        assert w.owns(token) and not errors
        w.run(w.spark('control','cluster','sync','--revision',commit,'--apply'),out,'sync-apply')
        if options.kernel_jobs:
            source = HERE.parent.parent/'2026-11-04-semantic-indexer/run_window.py'
            spec = importlib.util.spec_from_file_location('isolated_kernels',source)
            kernels = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernels)
            kernels.HERE = HERE.parent
            w.run(w.spark('restore','cluster','stop'),out,'kernel-stop-plan')
            w.run(w.spark('restore','cluster','stop','--apply'),out,'kernel-stop-apply')
            with open('/tmp/spark-lab-gpu.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                for job in options.kernel_jobs:
                    requests = w.each(w.HOLD_CODE,{'action':'read'})
                    assert not any(x['spark-request.json'] or x['spark3-request.json'] for x in requests.values())
                    began = time.monotonic()
                    while kernels.thermal() >= 55:
                        assert w.owns(token) and not errors and time.monotonic()-began < 600
                        time.sleep(5)
                    kernels.kernel(job,out,token,errors)

        completed = []
        jobs = ([tuple(job.split(':',1)) for job in (options.quality_profiles or options.diagnostic_profiles)]
                if options.quality_profiles or options.diagnostic_profiles else
                [('control-a','control'),('shortlist-a',options.candidate_profile),('shortlist-b',options.candidate_profile),('control-b','control')])
        (out/'jobs.json').write_text(json.dumps({'jobs':jobs,'diagnostic_timing':bool(options.diagnostic_profiles),
            'rule':'timing is independent; token mismatches prohibit acceptance'},indent=2)+'\n')
        for label,profile in jobs:
            assert len((label,profile)) == 2 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label+profile)
            assert (HERE/(profile+'.json')).is_file()
            assert w.owns(token) and not errors
            requests = w.each(w.HOLD_CODE,{'action':'read'})
            assert not any(x['spark-request.json'] or x['spark3-request.json'] for x in requests.values()),'pending requester'
            w.start(profile,out,'boot-'+label)
            w.run([sys.executable,str(HERE/'request_screen.py'),'corpus','--private',str(private)],out,'corpus-'+label)
            if label == 'control-a':
                (out/'corpus-manifest.json').write_text((private/'manifest.json').read_text())
                (out/'identity.json').write_text((private/'identity.json').read_text())
            capture = private/((out.name+'-'+label if options.quality_profiles else label)+'.json')
            w.run([sys.executable,str(HERE.parent/'token_gate.py'),'capture',
                   '--url','http://127.0.0.1:8001','--requests',str(private/'requests.jsonl'),
                   '--identity',str(private/'identity.json'),'--out',str(capture)],out,'tokens-'+label)
            if label == 'control-a' and options.reference_private:
                w.run([sys.executable,str(HERE.parent/'token_gate.py'),'compare',
                       str(Path(options.reference_private)/'control-a.json'),str(capture)],out,'native-baseline-repeat',
                       allow_failure=bool(options.diagnostic_profiles))
            if label != 'control-a':
                # Failed presets stay failed. A declared token-only calibration
                # may continue to its next preset; ordinary qualification stops.
                w.run([sys.executable,str(HERE.parent/'token_gate.py'),'compare',
                       str(private/'control-a.json'),str(capture)],out,'token-comparison-'+label,
                       allow_failure=bool(options.diagnostic_profiles or (options.quality_profiles and profile != 'control')))
                if options.quality_profiles or options.diagnostic_profiles:
                    verdict = json.loads((out/('token-comparison-'+label+'.txt')).read_text())
                    assert 'passed' in verdict and verdict['total_requests'] == 9
            if not options.quality_profiles:
                parallel = private/(label+'-c4.json')
                w.run([sys.executable,str(HERE.parent/'token_gate.py'),'capture',
                       '--url','http://127.0.0.1:8001','--requests',str(private/'requests.jsonl'),
                       '--identity',str(private/'identity.json'),'--concurrency','4','--out',str(parallel)],out,'tokens-'+label+'-c4')
                if label != 'control-a':
                    w.run([sys.executable,str(HERE.parent/'token_gate.py'),'compare',
                           str(private/'control-a-c4.json'),str(parallel)],out,'token-comparison-'+label+'-c4',
                           allow_failure=bool(options.diagnostic_profiles))
                else:
                    repeat = private/'control-a-c4-repeat.json'
                    w.run([sys.executable,str(HERE.parent/'token_gate.py'),'capture',
                           '--url','http://127.0.0.1:8001','--requests',str(private/'requests.jsonl'),
                           '--identity',str(private/'identity.json'),'--concurrency','4','--out',str(repeat)],out,'tokens-control-a-c4-repeat')
                    w.run([sys.executable,str(HERE.parent/'token_gate.py'),'compare',
                           str(parallel),str(repeat)],out,'native-parallel-repeat',allow_failure=bool(options.diagnostic_profiles))
                # Only explicit token mismatches may continue diagnostic timing.
                # Missing IDs, mismatched inputs or malformed comparisons are errors.
                for comparison in out.glob('token-comparison-'+label+'*.txt'):
                    verdict = json.loads(comparison.read_text())
                    assert type(verdict.get('passed')) is bool and verdict.get('total_requests') == 9
                for stem in (['native-baseline-repeat','native-parallel-repeat'] if label == 'control-a' else []):
                    comparison = out/(stem+'.txt')
                    if comparison.exists():
                        verdict = json.loads(comparison.read_text())
                        assert type(verdict.get('passed')) is bool and verdict.get('total_requests') == 9
                w.run([sys.executable,str(HERE/'measure_requests.py'),'--profile',w.REL+'/'+profile+'.json',
                       '--private',str(private),'--out',str(out/('requests-'+label+'.json'))],out,'bench-'+label)
            completed.append(label)
            (out/'completed.json').write_text(json.dumps({'at':w.stamp(),'completed':completed},indent=2)+'\n')
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        try:
            w.restore(token,out,'final-restore',finish)
        finally:
            finish()


if __name__ == '__main__':
    main()
