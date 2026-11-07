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
    parser.add_argument('--candidate-config',default=PROFILE_ROOT+'/aggressive-fast.json')
    parser.add_argument('--native-config',default=PROFILE_ROOT+'/control.json')
    parser.add_argument('--controls-from',type=Path,help='published same-day native receipts with identical model and client inputs')
    parser.add_argument('--watchdog')
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'spark-0013':
        raise SystemExit('only the owner-authorized isolated sp5-sp7 group')
    out = args.out.resolve()
    if args.watchdog:
        return w.watchdog(args.watchdog,out)
    assert args.cache
    private_root = Path.home()/'spark-lab/private/answer-quality'/out.relative_to(Path.home()/'spark-lab/results')
    assert not private_root.exists(),'private output already exists; use a new complete result path'
    from prepare import LOCK,digest
    assert digest(args.cache/'selection.json') == LOCK['selection_sha256']
    selection = json.loads((args.cache/'selection.json').read_text())
    assert selection['protocol'] == LOCK['protocol']
    assert digest(args.cache/(args.split+'-requests.jsonl')) == selection['artifact_sha256'][args.split+'-requests.jsonl']
    assert not out.exists();out.mkdir(parents=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','branch','-r','--contains',commit],cwd=ROOT,text=True).strip()
    candidate = Path(args.candidate_config)
    assert not candidate.is_absolute() and '..' not in candidate.parts and (ROOT/candidate).is_file()
    native_config = Path(args.native_config)
    assert not native_config.is_absolute() and '..' not in native_config.parts and (ROOT/native_config).is_file()
    if args.controls_from:
        assert str(native_config) == PROFILE_ROOT+'/control.json', 'custom native profile requires fresh controls'
        import datetime
        import hashlib
        reference = (ROOT/args.controls_from).resolve()
        relative = reference.relative_to(ROOT)
        subprocess.run(['git','cat-file','-e','HEAD:'+str(relative/'restored.json')],cwd=ROOT,check=True)
        restored = json.loads((reference/'restored.json').read_text())
        assert restored['restored'] and restored['holds_removed']
        recorded = json.loads((reference/'source.json').read_text())
        assert recorded['split'] == args.split and recorded['sources_lock'] == LOCK
        unchanged = ['config/cluster.json',
            'experiments/2026-11-04-token-lsa/paged/control.json',
            'experiments/2026-11-04-clustered-indexer/serving/restore.json',
            'experiments/2026-11-04-clustered-indexer/replay/overlay/indexer.py',
            'experiments/2026-11-04-token-lsa/bundle/overlay/vllm/models/deepseek_v4_1/tilelang/shortlist_indexer.py',
            'experiments/2026-11-05-answer-quality/evaluate.py',
            'experiments/2026-11-05-answer-quality/api_metrics.py',
            'experiments/2026-11-05-answer-quality/measure.py',
            'experiments/2026-11-05-answer-quality/prepare.py']
        unchanged += [str(p.relative_to(ROOT)) for p in (ROOT/PROFILE_ROOT/'overlay').glob('*.py')]
        for name in unchanged:
            old = subprocess.check_output(['git','show',recorded['commit']+':'+name],cwd=ROOT)
            assert old == (ROOT/name).read_bytes(), name+' changed; fresh native controls required'
        refs = {}
        for label in ('native-a','native-b'):
            receipt = json.loads((reference/('benchmark-'+label+'-exit.json')).read_text())
            assert receipt['exit'] == 0 and receipt['started_at'][:10] == datetime.datetime.now(datetime.timezone.utc).date().isoformat()
            answer = json.loads((reference/('answers-'+label+'.json')).read_text())
            assert answer['complete'] and answer['selection_sha256'] == LOCK['selection_sha256']
            for suffix in ('.json','-conditions.json'):
                name = 'answers-'+label+suffix
                raw = (reference/name).read_bytes()
                (out/name).write_bytes(raw)
                refs[name] = hashlib.sha256(raw).hexdigest()
        (out/'reference-controls.json').write_text(json.dumps({'source':str(relative),'files':refs},indent=2)+'\n')
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (out/'source.json').write_text(json.dumps({'commit':commit,'split':args.split,'native_config':str(native_config),'candidate_config':str(candidate),'sources_lock':json.loads((HERE/'sources.lock.json').read_text())},indent=2)+'\n')
    w.run([sys.executable,str(source),'--out',str(out/'preflight'),'--preflight-only'],out,'preflight')
    for label,config in [('native',native_config),('candidate',candidate)]:
        previous = w.REL
        try:
            w.REL = str(config.parent)
            w.run(w.spark(config.stem,'doctor'),out,'preflight-'+label+'-profile')
        finally:w.REL = previous
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
        jobs = [('native-a','control'),('lsa-a',candidate.stem),('lsa-b',candidate.stem),('native-b','control')]
        (out/'jobs.json').write_text(json.dumps({'jobs':jobs,'native_config':str(native_config),'candidate_config':str(candidate),'controls_from':str(args.controls_from) if args.controls_from else None,'split':args.split,'rule':'independent final-answer grading; no native output oracle'},indent=2)+'\n')
        completed = ['native-a','native-b'] if args.controls_from else []
        for label,profile in jobs:
            if args.controls_from and label.startswith('native-'):
                continue
            assert w.owns(token) and not errors
            holds = w.each(w.HOLD_CODE,{'action':'read'})
            assert not any(h['spark-request.json'] or h['spark3-request.json'] for h in holds.values()),'pending requester'
            previous = w.REL
            config = candidate if label.startswith('lsa-') else native_config
            try:
                w.REL = str(config.parent)
                w.start(config.stem,out,'boot-'+label)
            finally:
                w.REL = previous
            private = private_root/label
            w.run([sys.executable,str(HERE/'measure.py'),'--profile',str(config),
                   '--cache',str(args.cache),'--split',args.split,'--private',str(private),
                   '--out',str(out/('answers-'+label+'.json'))],out,'benchmark-'+label)
            if label.startswith('lsa-'):
                from summarize import paired
                base_answer = json.loads((out/'answers-native-a.json').read_text())
                answer = json.loads((out/('answers-'+label+'.json')).read_text())
                assert answer.get('complete') and base_answer.get('complete')
                assert answer['protocol'] == base_answer['protocol'] and answer['selection_sha256'] == base_answer['selection_sha256']
                assert len(answer['phases']) == len(base_answer['phases']) > 0
                checks = []
                for old,new in zip(base_answer['phases'],answer['phases']):
                    quality = paired(old['records'],new['records'])
                    native_rows={r['id']:r for r in old['records']}
                    incomplete = [r['id'] for r in new['records']
                                  if (r['status'] != 'complete' or r.get('finish_reason') != 'stop')
                                  and native_rows[r['id']]['status']=='complete'
                                  and native_rows[r['id']].get('finish_reason')=='stop']
                    checks.append({'phase':new['name'],'losses':quality['losses'],'incomplete':incomplete})
                if any(c['losses'] or c['incomplete'] for c in checks):
                    (out/('quality-reject-'+label+'.json')).write_text(json.dumps({'passed':False,'checks':checks,'scope':'early observed-regression rejection; not a two-boot performance comparison'},indent=2)+'\n')
                    raise RuntimeError('candidate final-answer quality regression; restore before further benchmarks')
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
