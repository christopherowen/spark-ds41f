#!/usr/bin/env python3
"""Independently read C36 identity only after C42's actual successful wait exit."""
import json
import subprocess

PROGRAM = r'''
import concurrent.futures,datetime,json,os,pathlib,shlex,subprocess
root=pathlib.Path('/home/swank/projects/spark-ds41f');os.chdir(root)
setup=pathlib.Path('/home/swank/tl-logs/compiler-live-qualification/setup')
control=root/'results/private/lab/candidate-42-round/receipt.json'
r=json.loads(control.read_text())
ep=setup/'exit.json'
e=json.loads(ep.read_text()) if ep.exists() else None
if not (r.get('completed') and r.get('restored') and r.get('ended') and
        r.get('commit')=='b6896b389c5bb0004e13fe9f4f941a31a60f87ca' and
        not r.get('error') and not r.get('restore_error') and
        not r.get('management_loss_latched') and not r.get('management_unreachable') and
        e and e.get('actual_wait_exit')==0):
 print(json.dumps({'ready':False,'reason':'native execution has not successfully completed/restored with actual wait exit0','phase':r.get('phase'),'checks_run':False}));raise SystemExit(0)
launch=json.loads((setup/'launch.json').read_text())
states={}
for key in ('controller_pid','supervisor_pid'):
 pid=launch[key]
 try:
  os.kill(pid,0);states[str(pid)]=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]
 except (ProcessLookupError,FileNotFoundError):states[str(pid)]='absent'
if any(state not in ('absent','Z') for state in states.values()):
 print(json.dumps({'ready':False,'reason':'native controller or supervisor remains active','states':states,'checks_run':False}));raise SystemExit(0)
from spark_ds41f import config
restore='experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json'
image='sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11'
name=config.read_cluster(restore)['container']['name']
def rank(host):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(['docker','inspect',name])],capture_output=True,text=True,timeout=20)
 if p.returncode:return {'host':host,'exit':p.returncode,'error':p.stderr}
 x=json.loads(p.stdout)[0]
 return {'host':host,'container_id':x['Id'],'image':x['Image'],'running':x['State']['Running'],'status':x['State']['Status'],'started_at':x['State']['StartedAt'],'restart_count':x['RestartCount'],'profile':x['Config'].get('Labels',{}).get('spark.cluster-config')}
def ranks():
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:return list(pool.map(rank,['dgx1','dgx2','dgx3']))
before=ranks()
argv=['python3','bin/spark','--cluster-config',restore,'doctor','--live']
p=subprocess.run(argv,capture_output=True,text=True,timeout=120)
doctor={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
after=ranks()
shared={}
for key,n in [('hold','spark-hold.json'),('request','spark-request.json'),('legacy_request','spark3-request.json')]:
 path=pathlib.Path.home()/n;shared[key]=json.loads(path.read_text()) if path.exists() else None
passed=p.returncode==0 and {x['host'] for x in after}=={'dgx1','dgx2','dgx3'} and before==after and all(x.get('running') is True and x.get('restart_count')==0 and x.get('image')==image and x.get('profile')==restore for x in after)
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ranks':after,'ranks_before_doctor':before,'same_containers_across_independent_check':before==after,'process_states':states,'native_actual_exit':e,'shared':shared,'passed':passed}
record={'doctor':doctor,'restoration':result,'no_node_or_serving_changes':True}
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
with (setup/('independent-restoration-check-'+stamp+'.json')).open('x') as f:json.dump(record,f,indent=2);f.write('\n')
if passed:
 for filename,data in [('post-round-restore-doctor.json',doctor),('post-round-restoration-ranks.json',result)]:
  with (setup/filename).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
print(json.dumps({'ready':True,'checks_run':True,'passed':passed,'doctor_exit':doctor['exit'],'ranks':after,'process_states':states,'shared':shared}))
'''


def main():
    compile(PROGRAM, "remote-independent-restoration-read", "exec")
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx1", "python3", "-"],
        input=PROGRAM, capture_output=True, text=True, timeout=180,
    )
    if p.returncode:
        raise SystemExit(p.stderr)
    result = json.loads(p.stdout)
    print(json.dumps(result))
    if not result["ready"]:
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
