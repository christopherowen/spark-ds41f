"""Owner-authorized paired request window on idle Spaceport sp5-sp7.

Uses only coordinated, named-profile spark operations. Original production
checkouts remain untouched. Independent token-scoped watchdog restores C37 if
this controller stops refreshing the three native holds for fifteen minutes.
"""
import argparse
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[3]
REL='experiments/2026-11-04-clustered-indexer/serving'
NODES=['10.30.0.13','10.30.0.14','10.30.0.15']
IMAGE='sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a'
HOLDER='clustered-indexer-request-window'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def remote(node,program,data=None):
    code='DATA='+repr(data)+'\n'+program
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',
                           'chris@'+node,'python3 -'],input=code,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(node+': '+result.stderr[-2000:])
    return json.loads(result.stdout)


def each(program,data=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(lambda node:remote(node,program,data),NODES))
    return dict(zip(NODES,values))


HOLD_CODE="""from pathlib import Path
import json,os
p=Path.home()/'spark-hold.json'
if DATA['action']=='read':
 print(json.dumps({n:json.loads((Path.home()/n).read_text()) if (Path.home()/n).exists() else None
                   for n in ['spark-hold.json','spark3-hold.json','spark-request.json','spark3-request.json']}))
elif DATA['action']=='open':
 assert not (Path.home()/'spark3-hold.json').exists()
 assert not (Path.home()/'spark-request.json').exists()
 with p.open('x') as handle:json.dump(DATA['hold'],handle,indent=2)
 print('true')
else:
 current=json.loads(p.read_text());assert current['token']==DATA['token']
 if DATA['action']=='refresh':
  current['heartbeat']=DATA['heartbeat'];tmp=p.with_name('.clustered-hold-'+DATA['token'])
  tmp.write_text(json.dumps(current,indent=2)+'\\n');os.replace(tmp,p)
 elif DATA['action']=='remove':p.unlink()
 print('true')
"""


def owns(token):
    holds=each(HOLD_CODE,{'action':'read'})
    return all(x['spark-hold.json'] and x['spark-hold.json'].get('token')==token
               and not x['spark3-hold.json'] for x in holds.values())


def run(command,out,stem,*,allow_failure=False):
    (out/(stem+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
    started=stamp();began=time.monotonic()
    with (out/(stem+'.txt')).open('w') as handle:
        result=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    receipt={'argv':command,'exit':result.returncode,'started_at':started,'ended_at':stamp(),
             'seconds':time.monotonic()-began}
    (out/(stem+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'step':stem,**receipt}),flush=True)
    if result.returncode and not allow_failure:raise RuntimeError(stem+' failed; see native receipt')
    return result.returncode


def spark(profile,*args):return ['bin/spark','--cluster-config',REL+'/'+profile+'.json',*args]


def start(profile,out,stem):
    command=spark(profile,'cluster','start','--replace')
    run(command,out,stem+'-plan')
    plan=(out/(stem+'-plan.txt')).read_text()
    live=next((x for x in plan.splitlines() if x.startswith('live:')),None)
    shape=next((x for x in plan.splitlines() if x.startswith('shape:')),None)
    assert live and shape and ('unchanged' in shape or 'new engine' in shape),(live,shape)
    assert '--apply would refuse:' not in plan,plan
    # Retain the exact live/shape lines read before every start apply.
    (out/(stem+'-review.json')).write_text(json.dumps({'live':live,'shape':shape},indent=2)+'\n')
    run(command+['--apply'],out,stem+'-apply')
    run(spark(profile,'doctor','--live'),out,stem+'-doctor')


def restore(token,out,stem):
    if not owns(token):raise RuntimeError('reservation changed; refusing to overwrite another holder')
    # If no serving mutation occurred, avoid an unnecessary C37 restart.
    code=run(spark('restore','doctor','--live'),out,stem+'-already-live',allow_failure=True)
    if code:start('restore',out,stem)
    check=each("""import subprocess,json
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True))[0]
p=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/chris/projects/spark-ds41f',text=True).strip()
print(json.dumps({'image':x['Image'],'running':x['State']['Running'],'original_checkout':p,
 'experiment_mounts':[m['Destination'] for m in x['Mounts'] if 'clustered' in m['Source']],
 'search_env':[e for e in x['Config']['Env'] if e.startswith('VLLM_DS41_CLUSTERED_SEARCH=')]}))
""")
    assert all(x['image']==IMAGE and x['running'] and not x['experiment_mounts'] and not x['search_env'] for x in check.values())
    (out/(stem+'-verification.json')).write_text(json.dumps(check,indent=2)+'\n')
    each(HOLD_CODE,{'action':'remove','token':token})
    (out/'restored.json').write_text(json.dumps({'at':stamp(),'restored':True,'holds_removed':True},indent=2)+'\n')


def watchdog(token,out):
    while not (out/'restored.json').exists():
        time.sleep(15)
        holds=each(HOLD_CODE,{'action':'read'})
        owned=[x['spark-hold.json'] for x in holds.values() if x['spark-hold.json'] and x['spark-hold.json'].get('token')==token]
        if not owned:return
        age=max((dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(h['heartbeat'])).total_seconds() for h in owned)
        if age>900:
            restore(token,out,'watchdog-restore');return


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--watchdog')
    args=parser.parse_args();out=Path(args.out).resolve()
    if socket.gethostname()!='spark-0013':raise SystemExit('only the idle Spaceport sp5-sp7 group is authorized')
    if args.watchdog:return watchdog(args.watchdog,out)
    out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','branch','-r','--contains',commit],cwd=ROOT,text=True).strip()
    holds=each(HOLD_CODE,{'action':'read'})
    assert not any(any(v.values()) for v in holds.values()),holds
    run(spark('restore','doctor','--live'),out,'preflight-doctor')
    # Require thirty seconds with no active/waiting work and no completion.
    baseline=None
    for i in range(7):
        metrics=urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=15).read().decode()
        lines=[x for x in metrics.splitlines() if not x.startswith('#') and any(k in x for k in
               ['num_requests_running{','num_requests_waiting{','request_success_total{'])]
        for line in lines:
            if 'num_requests_' in line:assert float(line.split()[-1])==0,line
        successes=[x for x in lines if 'request_success_total{' in x]
        if baseline is None:baseline=successes
        assert successes==baseline,'serving traffic changed during idle guard'
        (out/f'idle-{i}.json').write_text(json.dumps({'at':stamp(),'metrics':lines},indent=2)+'\n')
        if i<6:time.sleep(5)
    token=uuid.uuid4().hex
    hold={'holder':HOLDER,'token':token,'since':stamp(),'heartbeat':stamp(),'expected_end':None,
          'nodes':['sp5','sp6','sp7'],'rule':'exclusive paired LSA request measurements; restore recorded C37 once at end',
          'commit':commit,'production_config':REL+'/restore.json','controller_output':str(out)}
    acquired=[]
    try:
        for node in NODES:
            remote(node,HOLD_CODE,{'action':'open','hold':hold});acquired.append(node)
    except Exception:
        for node in acquired:remote(node,HOLD_CODE,{'action':'remove','token':token})
        raise
    (out/'window.json').write_text(json.dumps(hold,indent=2)+'\n')
    stop=threading.Event();heartbeat_errors=[]
    def beat():
        while not stop.wait(30):
            try:each(HOLD_CODE,{'action':'refresh','token':token,'heartbeat':stamp()})
            except Exception as error:heartbeat_errors.append(str(error));return
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    with (out/'watchdog.txt').open('w') as handle:
        subprocess.Popen([sys.executable,__file__,'--out',str(out),'--watchdog',token],cwd=ROOT,
                         stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    try:
        assert owns(token)
        command=spark('control','cluster','sync','--revision',commit)
        run(command,out,'sync-plan');run(command+['--apply'],out,'sync-apply')
        completed=[]
        for label,profile in [('lsa-a','candidate'),('control-a','control'),('control-b','control'),('lsa-b','candidate')]:
            assert owns(token) and not heartbeat_errors
            requests=each(HOLD_CODE,{'action':'read'})
            if any(x['spark-request.json'] or x['spark3-request.json'] for x in requests.values()):
                raise RuntimeError('pending requester: return group after current completed job')
            start(profile,out,'boot-'+label)
            run(spark(profile,'bench','--suites','quality,decode,prefill','--url','http://127.0.0.1:8001',
                '--compare','none','--output',str(out/label),'--concurrency','1,4,8','--decode-cases','prose,code',
                '--min-samples','3','--max-samples','4','--prefill-sizes','32768,131072','--prefill-repeats','3',
                '--prefill-text','source'),out,'bench-'+label)
            completed.append(label)
            (out/'completed.json').write_text(json.dumps({'completed':completed,'at':stamp()},indent=2)+'\n')
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc());raise
    finally:
        # Keep the heartbeat alive throughout coordinated restoration.
        try:restore(token,out,'final-restore')
        finally:stop.set();thread.join(timeout=35)


if __name__=='__main__':main()
