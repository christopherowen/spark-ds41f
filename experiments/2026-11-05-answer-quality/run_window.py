"""Coordinated independent-answer benchmark on the isolated sp5-sp7 group."""
import argparse
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
ROOT = HERE.parents[1]
source = HERE.parent/'2026-11-04-token-lsa/paged/run_serving.py'
spec = importlib.util.spec_from_file_location('page_window',source)
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)
w = page.w
PROFILE_ROOT = 'experiments/2026-11-04-token-lsa/paged'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--split',choices=['pilot','heldout'],default='pilot')
    parser.add_argument('--watchdog')
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'spark-0013':
        raise SystemExit('only the owner-authorized isolated sp5-sp7 group')
    out = args.out.resolve()
    if args.watchdog:
        return w.watchdog(args.watchdog,out)
    assert args.cache
    from prepare import LOCK,digest
    assert digest(args.cache/'selection.json') == LOCK['selection_sha256']
    selection = json.loads((args.cache/'selection.json').read_text())
    assert selection['protocol'] == LOCK['protocol']
    assert digest(args.cache/(args.split+'-requests.jsonl')) == selection['artifact_sha256'][args.split+'-requests.jsonl']
    assert not out.exists();out.mkdir(parents=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','branch','-r','--contains',commit],cwd=ROOT,text=True).strip()
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (out/'source.json').write_text(json.dumps({'commit':commit,'split':args.split,'sources_lock':json.loads((HERE/'sources.lock.json').read_text())},indent=2)+'\n')
    w.run([sys.executable,str(source),'--out',str(out/'preflight'),'--preflight-only'],out,'preflight')
    if args.preflight_only:
        return
    baseline = None
    for i in range(7):
        metrics = urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=15).read().decode()
        lines = [line for line in metrics.splitlines() if not line.startswith('#') and any(k in line for k in
                 ('num_requests_running{','num_requests_waiting{','request_success_total{'))]
        assert all(float(line.split()[-1]) == 0 for line in lines if 'num_requests_' in line)
        successes = [line for line in lines if 'request_success_total{' in line]
        if baseline is None:
            baseline = successes
        assert baseline == successes,'serving traffic changed during idle guard'
        (out/f'idle-{i}.json').write_text(json.dumps({'at':w.stamp(),'metrics':lines},indent=2)+'\n')
        if i < 6:
            time.sleep(5)
    token = uuid.uuid4().hex
    hold = {'holder':'independent-answer-quality-window','token':token,'since':w.stamp(),
            'heartbeat':w.stamp(),'expected_end':None,'nodes':['sp5','sp6','sp7'],'commit':commit,
            'rule':'exclusive complete-answer benchmark; restore tracked C37 once',
            'production_config':PROFILE_ROOT+'/restore.json','controller_output':str(out)}
    acquired = []
    try:
        for node in w.NODES:
            w.remote(node,w.HOLD_CODE,{'action':'open','hold':hold});acquired.append(node)
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
                errors.append(str(error));return
    thread = threading.Thread(target=beat,daemon=True);thread.start()
    def finish():
        stop.set();thread.join(timeout=35)
        assert not thread.is_alive(),'heartbeat writer remains active; retaining leases'
    with (out/'watchdog.txt').open('w') as handle:
        subprocess.Popen([sys.executable,__file__,'--out',str(out),'--watchdog',token],cwd=ROOT,
                         stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        assert w.owns(token) and not errors
        w.run(w.spark('control','cluster','sync','--revision',commit),out,'sync-plan')
        w.run(w.spark('control','cluster','sync','--revision',commit,'--apply'),out,'sync-apply')
        jobs = [('native-a','control'),('lsa-a','aggressive-fast'),('lsa-b','aggressive-fast'),('native-b','control')]
        (out/'jobs.json').write_text(json.dumps({'jobs':jobs,'split':args.split,'rule':'independent final-answer grading; no native output oracle'},indent=2)+'\n')
        completed = []
        for label,profile in jobs:
            assert w.owns(token) and not errors
            holds = w.each(w.HOLD_CODE,{'action':'read'})
            assert not any(h['spark-request.json'] or h['spark3-request.json'] for h in holds.values()),'pending requester'
            w.start(profile,out,'boot-'+label)
            private = Path.home()/'spark-lab/private/answer-quality'/out.name/label
            w.run([sys.executable,str(HERE/'measure.py'),'--profile',PROFILE_ROOT+'/'+profile+'.json',
                   '--cache',str(args.cache),'--split',args.split,'--private',str(private),
                   '--out',str(out/('answers-'+label+'.json'))],out,'benchmark-'+label)
            completed.append(label)
            (out/'completed.json').write_text(json.dumps({'at':w.stamp(),'completed':completed},indent=2)+'\n')
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc());raise
    finally:
        try:
            w.restore(token,out,'final-restore',finish)
        finally:
            finish()


if __name__ == '__main__':
    main()
