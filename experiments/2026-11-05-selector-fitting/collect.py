"""Reserved all-node collection of native operands; never a timed benchmark."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import threading
import traceback
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGE = ROOT/'experiments/2026-11-04-token-lsa/paged/run_serving.py'
spec = importlib.util.spec_from_file_location('capture_window',PAGE)
page = importlib.util.module_from_spec(spec);spec.loader.exec_module(page)
w = page.w
CAPTURE = Path.home()/'projects/spark-ds41f/cache/selector-fitting-20261105'

ARM = '''import json,sys
from pathlib import Path
x=json.load(sys.stdin);p=Path.home()/'projects/spark-ds41f/cache/selector-fitting-20261105'
if x['action']=='check':
 assert not p.exists() or not list(p.iterdir()),'capture directory is not empty'
 p.mkdir(parents=True,exist_ok=True)
elif x['action']=='arm':
 with (p/'arm.json').open('x') as f:json.dump(x['case'],f)
elif x['action']=='disarm':
 (p/'arm.json').unlink(missing_ok=True)
elif x['action']=='manifest':
 import hashlib
 files={a.name:{'bytes':a.stat().st_size,'sha256':hashlib.file_digest(a.open('rb'),'sha256').hexdigest()} for a in p.glob('*.pt')}
 assert len(files)==12 and not list(p.glob('*.tmp')) and not (p/'arm.json').exists(),files
 print(json.dumps(files));raise SystemExit
print(json.dumps({'ok':True}))
'''


def request(route,body=None,timeout=30):
    raw = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:8001'+route,data=raw,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def prompts():
    # Documents, not neighboring query rows, define this split. Identical source
    # files are deduplicated before assigning each entire file to one cohort.
    source = Path(sysconfig.get_paths()['stdlib'])
    files = {}
    for path in sorted(source.glob('*.py')):
        if path.stat().st_size > 4096:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            files.setdefault(sha,path)
    ordered = sorted(files,key=lambda s:hashlib.sha256(('selector-fitting-20261105:'+s).encode()).hexdigest())
    result = []
    for i,name in enumerate(('fit-a','fit-b','validation-a')):
        parts,manifest = [],[]
        for sha in ordered[i::3]:
            path = files[sha]
            parts.append(path.read_text(encoding='utf-8',errors='replace'))
            manifest.append({'file':path.name,'sha256':sha,'bytes':path.stat().st_size})
            if sum(map(len,parts)) >= 131072*3.8:
                break
        assert sum(map(len,parts)) >= 131072*3.8,'not enough disjoint source text'
        text = 'Document '+name+':\n'+''.join(parts)
        result.append({'case_id':name,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                       'prompt':text,'files':manifest})
    assert len({f['sha256'] for r in result for f in r['files']}) == sum(len(r['files']) for r in result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--watchdog')
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'spark-0013'
    out = args.out.resolve()
    if args.watchdog:
        return w.watchdog(args.watchdog,out)
    assert not out.exists();out.mkdir(parents=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','branch','-r','--contains',commit],cwd=ROOT,text=True).strip()
    cases = prompts()
    (out/'source.json').write_text(json.dumps({'commit':commit,'cases':[{k:v for k,v in r.items() if k!='prompt'} for r in cases],
        'purpose':'untimed native operand collection; no benchmark questions or fitted maps'},indent=2)+'\n')
    w.run([sys.executable,str(PAGE),'--out',str(out/'preflight'),'--preflight-only'],out,'preflight')
    if args.preflight_only:return
    # Require a quiet restored service before taking its separate Spaceport
    # reservations. A queued user request yields after the current capture.
    baseline = None
    import time
    for i in range(7):
        text = urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=15).read().decode()
        lines = [s for s in text.splitlines() if not s.startswith('#') and any(k in s for k in ('num_requests_running{','num_requests_waiting{','request_success_total{'))]
        assert all(float(s.split()[-1]) == 0 for s in lines if 'num_requests_' in s)
        successes = [s for s in lines if 'request_success_total{' in s]
        if baseline is None:baseline = successes
        assert successes == baseline
        (out/f'idle-{i}.json').write_text(json.dumps({'at':w.stamp(),'metrics':lines},indent=2)+'\n')
        if i<6:time.sleep(5)
    token = uuid.uuid4().hex
    hold = {'holder':'native-selector-fitting-capture','token':token,'since':w.stamp(),'heartbeat':w.stamp(),
        'expected_end':None,'nodes':['sp5','sp6','sp7'],'commit':commit,'rule':'exclusive untimed native operand collection; restore C37 once',
        'production_config':'experiments/2026-11-04-token-lsa/paged/restore.json','controller_output':str(out)}
    acquired = []
    try:
        for node in w.NODES:
            w.remote(node,w.HOLD_CODE,{'action':'open','hold':hold});acquired.append(node)
    except Exception:
        for node in acquired:w.remote(node,w.HOLD_CODE,{'action':'remove','token':token})
        raise
    (out/'window.json').write_text(json.dumps(hold,indent=2)+'\n')
    stop,errors = threading.Event(),[]
    def beat():
        while not stop.wait(30):
            try:w.each(w.HOLD_CODE,{'action':'refresh','token':token,'heartbeat':w.stamp()})
            except Exception as error:errors.append(str(error));return
    thread = threading.Thread(target=beat,daemon=True);thread.start()
    def finish():
        stop.set();thread.join(35);assert not thread.is_alive()
    with (out/'watchdog.txt').open('w') as handle:
        subprocess.Popen([sys.executable,__file__,'--out',str(out),'--watchdog',token],cwd=ROOT,
                         stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    private = Path.home()/'spark-lab/private/selector-fitting'/out.name
    try:
        assert w.owns(token) and not errors
        w.run(w.spark('control','cluster','sync','--revision',commit),out,'sync-plan')
        w.run(w.spark('control','cluster','sync','--revision',commit,'--apply'),out,'sync-apply')
        previous = w.REL
        try:
            w.REL = str(HERE.relative_to(ROOT));w.start('capture',out,'boot-capture')
        finally:w.REL = previous
        w.each(ARM,{'action':'check'})
        private.mkdir(mode=0o700,parents=True,exist_ok=False)
        model = request('/v1/models')['data'][0]['id'];assert model == 'deepseek-v4.1-flash'
        for case in cases:
            assert w.owns(token) and not errors
            holds = w.each(w.HOLD_CODE,{'action':'read'})
            assert not any(h['spark-request.json'] or h['spark3-request.json'] for h in holds.values()),'pending requester'
            (private/(case['case_id']+'-prompt.txt')).write_text(case['prompt'])
            tokens = request('/tokenize',{'model':model,'prompt':case['prompt'],'add_special_tokens':True},120)
            assert 2*(32769+256+64) <= int(tokens['count']) < int(tokens['max_model_len'])-1
            w.each(ARM,{'action':'arm','case':{k:case[k] for k in ('case_id','prompt_sha256')}})
            try:
                response = request('/v1/completions',{'model':model,'prompt':case['prompt'],'temperature':0,'seed':0,
                    'max_tokens':1,'ignore_eos':True,'stream':False,'cache_salt':uuid.uuid4().hex},1800)
            finally:w.each(ARM,{'action':'disarm'})
            (private/(case['case_id']+'-response.json')).write_text(json.dumps(response))
            (out/(case['case_id']+'.json')).write_text(json.dumps({'at':w.stamp(),'case_id':case['case_id'],
                'prompt_sha256':case['prompt_sha256'],'actual_prompt_tokens':tokens['count'],'usage':response['usage'],
                'response_sha256':hashlib.sha256(json.dumps(response).encode()).hexdigest(),'timed':False},indent=2)+'\n')
        manifest = w.each(ARM,{'action':'manifest'})
        (out/'snapshots.json').write_text(json.dumps(manifest,indent=2)+'\n')
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc());raise
    finally:
        try:
            try:w.each(ARM,{'action':'disarm'})
            finally:w.restore(token,out,'final-restore',finish)
        finally:finish()


if __name__ == '__main__':main()
