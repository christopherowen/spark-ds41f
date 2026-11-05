"""Owner-authorized matched quality window on idle Spaceport sp1-sp4."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('matrix_window',HERE.parent/'2026-11-04-clustered-indexer/serving/run_window.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
w.REL=str(HERE.relative_to(ROOT));w.NODES=['10.30.0.'+str(i) for i in range(9,13)]
SERVING='/home/chris/spark-lab/checkouts/lsa-serving'
IMAGE='sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'
from check_quality import canonical_check,check,early


def load_answer_module(name):
    spec=importlib.util.spec_from_file_location('matrix_answer_'+name,HERE.parent/'2026-11-05-answer-quality'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def quality_result(generations,grades,benchmark):
    if benchmark=='longbench':
        return load_answer_module('check_quality').check(
            [generations[n] for n in ('native-a','native-b')],
            [generations[n] for n in ('lsa-a','lsa-b')])
    return check(generations,grades,benchmark)


def early_result(generations,grades,label,benchmark):
    if benchmark=='longbench':
        failures=load_answer_module('watch_quality').regressions(generations['native-a'],generations[label])
        return {'passed':not failures,'checks':failures}
    losses=early(grades['native-a'],grades[label],benchmark)
    baseline={r['id']:r for r in generations['native-a']['records']}
    incomplete=[r['id'] for r in generations[label]['records']
                if (r['status']!='complete' or r['finish_reason']!='stop')
                and baseline[r['id']]['status']=='complete' and baseline[r['id']]['finish_reason']=='stop']
    return {'passed':not(losses or incomplete),'losses':losses,'incomplete':incomplete}


def profile_action(config,action,*args):
    previous=w.REL
    try:
        w.REL=str(config.parent)
        if action=='start':return w.start(config.stem,*args)
        return w.spark(config.stem,action,*args)
    finally:w.REL=previous


def runtime_identity():
    return w.each('''import json,subprocess
from pathlib import Path
x=json.loads(subprocess.check_output(['docker','inspect','spark-ds41f'],text=True))[0]
paths=['/home/chris/projects/spark-ds41f',DATA['serving']]
print(json.dumps({'image':x['Image'],'running':x['State']['Running'],
 'containers':subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines(),
 'checkouts':{p:{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
                'status':subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True)} for p in paths if Path(p).exists()},
 'experimental_mounts':[m['Destination'] for m in x['Mounts'] if '/experiments/' in m['Source']],
 'experimental_env':[e for e in x['Config']['Env'] if e.startswith(('VLLM_DS41_PAGE_','VLLM_DS41_CLUSTERED_SEARCH=','VLLM_DS41_SELECTOR_'))]}))
''',{'serving':SERVING})


def restore(token,out,stem,finish_heartbeat=None):
    assert w.owns(token),'reservation changed; refusing another holder'
    code=w.run(w.spark('restore','doctor','--live'),out,stem+'-already-live',allow_failure=True)
    if code:w.start('restore',out,stem)
    states=runtime_identity()
    assert all(s['image']==w.IMAGE and s['running'] and not s['experimental_mounts'] and not s['experimental_env']
               and all(not c['status'] for c in s['checkouts'].values()) for s in states.values()),states
    heads={s['checkouts'][SERVING]['head'] for s in states.values() if SERVING in s['checkouts']}
    assert len(heads)<=1,states
    (out/(stem+'-verification.json')).write_text(json.dumps(states,indent=2)+'\n')
    if finish_heartbeat:finish_heartbeat()
    w.each(w.HOLD_CODE,{'action':'remove','token':token})
    (out/'restored.json').write_text(json.dumps({'at':w.stamp(),'restored':True,'holds_removed':True},indent=2)+'\n')


w.restore=restore


def grade(cache,private,out,label):
    target=out/('grade-'+label);target.mkdir()
    uid=subprocess.check_output(['id','-u'],text=True).strip();gid=subprocess.check_output(['id','-g'],text=True).strip()
    command=['docker','run','--rm','--name','lsa-code-grader-'+label,'--user',uid+':'+gid,
        '--runtime=runc','--network','none','--read-only','--cpus','2','--memory','4g','--pids-limit','64',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--env','NVIDIA_VISIBLE_DEVICES=void',
        '--env','CUDA_VISIBLE_DEVICES=','--env','SPARK_EVALPLUS_ISOLATED=1','--env','OPENBLAS_NUM_THREADS=1',
        '--tmpfs','/tmp:rw,nosuid,size=128m','--mount','type=bind,source='+str(HERE.parent/'2026-11-05-answer-quality')+',target=/work,readonly',
        '--mount','type=bind,source='+str(cache)+',target=/data,readonly',
        '--mount','type=bind,source='+str(private)+',target=/samples,readonly',
        '--mount','type=bind,source='+str(target)+',target=/results','--entrypoint','python3',IMAGE,
        '/work/code_quality.py','grade','--cache','/data','--samples','/samples/samples.jsonl','--out','/results/receipt.json']
    w.run(command,out,'grade-'+label)
    return json.loads((target/'receipt.json').read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--benchmark',choices=['code','ruler','longbench'],default='code')
    p.add_argument('--cache',type=Path);p.add_argument('--split',choices=['pilot','heldout'],default='pilot')
    p.add_argument('--ruler-stage',choices=['full','screen'],default='full')
    p.add_argument('--candidate-config',type=Path,default=Path(w.REL)/'candidate.json')
    p.add_argument('--watchdog');p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();out=args.out.resolve()
    assert args.benchmark=='ruler' or args.ruler_stage=='full','ruler-stage applies only to RULER'
    assert socket.gethostname()=='spark-0009','only the authorized idle sp1-sp4 group'
    if args.watchdog:return w.watchdog(args.watchdog,out)
    assert args.cache
    assert args.cache.is_dir()
    candidate=args.candidate_config
    assert not candidate.is_absolute() and '..' not in candidate.parts and (ROOT/candidate).is_file()
    private=Path.home()/'spark-lab/private/quality-matrix'/out.relative_to(Path.home()/'spark-lab/results')
    assert not out.exists() and not private.exists()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','branch','-r','--contains',commit],cwd=ROOT,text=True).strip()
    assert not any(any(h.values()) for h in w.each(w.HOLD_CODE,{'action':'read'}).values())
    states=runtime_identity()
    assert all(s['image']==w.IMAGE and s['running'] and s['containers']==['spark-ds41f']
               and not s['experimental_mounts'] and not s['experimental_env']
               and all(not c['status'] for c in s['checkouts'].values()) for s in states.values()),states
    out.mkdir(parents=True)
    (out/'preflight-identity.json').write_text(json.dumps(states,indent=2)+'\n')
    (out/'source.json').write_text(json.dumps({'commit':commit,'benchmark':args.benchmark,
        'cache':str(args.cache),'split':args.split,'ruler_stage':args.ruler_stage,'nodes':w.NODES,'private':str(private),'candidate_config':str(candidate),
        'quality_gate_version':'observed-answer-and-completion-no-loss-v2',
        'rule':'same model and protocol; two fresh boots per arm; independent final-answer grading'},indent=2)+'\n')
    for profile in ('native','candidate','restore'):
        config=candidate if profile=='candidate' else Path(w.REL)/(profile+'.json')
        w.run(profile_action(config,'doctor'),out,'doctor-'+profile)
    if args.benchmark=='longbench':
        preparation=load_answer_module('prepare')
        assert preparation.digest(args.cache/'selection.json')==preparation.LOCK['selection_sha256']
        selection=json.loads((args.cache/'selection.json').read_text())
        assert selection['protocol']==preparation.LOCK['protocol']
        assert preparation.digest(args.cache/(args.split+'-requests.jsonl'))==selection['artifact_sha256'][args.split+'-requests.jsonl']
        extractor=preparation.grader(args.cache)
        assert extractor('The correct answer is (C).')=='C','pinned offline extractor preflight'
        (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    elif args.benchmark=='ruler':
        spec=importlib.util.spec_from_file_location('matrix_ruler_inputs',HERE.parent/'2026-11-05-context-quality/evaluate.py')
        ruler=importlib.util.module_from_spec(spec);spec.loader.exec_module(ruler)
        reference,manifest,cases,identity=ruler.load_inputs(args.cache,args.split,args.ruler_stage)
        (out/'input-validation.json').write_text(json.dumps({'manifest_sha256':identity,'cases':len(cases),
            'parent_manifest_sha256':reference['evaluation_sha256'],'stage':args.ruler_stage,'model_requests':0},indent=2)+'\n')
    baseline=None
    for i in range(7):
        metrics=urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=15).read().decode()
        lines=[s for s in metrics.splitlines() if not s.startswith('#') and any(k in s for k in ('num_requests_running{','num_requests_waiting{','request_success_total{'))]
        assert all(float(s.split()[-1])==0 for s in lines if 'num_requests_' in s)
        success=[s for s in lines if 'request_success_total{' in s]
        if baseline is None:baseline=success
        assert success==baseline,'serving traffic during idle guard'
        (out/('idle-'+str(i)+'.json')).write_text(json.dumps({'at':w.stamp(),'metrics':lines},indent=2)+'\n')
        if i<6:time.sleep(5)
    if args.preflight_only:return
    token=uuid.uuid4().hex
    hold={'holder':'lsa-independent-quality-window','token':token,'since':w.stamp(),'heartbeat':w.stamp(),
          'expected_end':None,'nodes':['sp1','sp2','sp3','sp4'],'commit':commit,
          'rule':'exclusive paired '+args.benchmark+' evaluation; restore tracked C37 once',
          'production_config':w.REL+'/restore.json','controller_output':str(out)}
    acquired=[]
    try:
        for node in w.NODES:w.remote(node,w.HOLD_CODE,{'action':'open','hold':hold});acquired.append(node)
    except Exception:
        for node in acquired:w.remote(node,w.HOLD_CODE,{'action':'remove','token':token})
        raise
    (out/'window.json').write_text(json.dumps(hold,indent=2)+'\n')
    stop=threading.Event();errors=[]
    def beat():
        while not stop.wait(30):
            try:w.each(w.HOLD_CODE,{'action':'refresh','token':token,'heartbeat':w.stamp()})
            except Exception as e:errors.append(str(e));return
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    def finish():
        stop.set();thread.join(timeout=35);assert not thread.is_alive(),'retaining leases while writer remains active'
    with (out/'watchdog.txt').open('x') as stream:
        subprocess.Popen([sys.executable,__file__,'--out',str(out),'--watchdog',token],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        assert w.owns(token) and not errors
        w.run(w.spark('restore','cluster','start'),out,'runtime-plan')
        plan=(out/'runtime-plan.txt').read_text()
        review={name:next((line for line in plan.splitlines() if line.startswith(name+':')),None) for name in ('live','shape')}
        assert all(review.values()) and 'unchanged' in review['shape'] and '--apply would refuse:' not in plan,review
        (out/'runtime-review.json').write_text(json.dumps(review,indent=2)+'\n')
        bootstrap=w.each('''import json,subprocess
from pathlib import Path
hold=json.loads((Path.home()/'spark-hold.json').read_text());assert hold['token']==DATA['token']
target=Path(DATA['serving']);created=not target.exists()
if created:
 target.parent.mkdir(parents=True,exist_ok=True)
 source=Path.home()/'projects/spark-ds41f'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
 subprocess.run(['git','clone',str(source),str(target)],check=True,stdout=subprocess.DEVNULL)
 subprocess.run(['git','remote','set-url','origin','https://github.com/christopherowen/spark-ds41f.git'],cwd=target,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=target,text=True).strip()
print(json.dumps({'created':created,'checkout':str(target),'head_before_coordinated_sync':subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()}))
''',{'serving':SERVING,'token':token})
        (out/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
        w.run(w.spark('restore','cluster','sync','--revision',commit),out,'sync-plan')
        w.run(w.spark('restore','cluster','sync','--revision',commit,'--apply'),out,'sync-apply')
        w.run(w.spark('restore','doctor','--live'),out,'preflight-live')
        if args.benchmark=='code':
            # Exercise every official canonical program before grading model output.
            # This CPU-only job runs before any timed serving arm.
            sys.path.insert(0,str(HERE.parent/'2026-11-05-answer-quality'))
            from code_quality import LOCK,inputs
            cases=inputs(args.cache);oracle=private/'canonical'
            oracle.mkdir(mode=0o700,parents=True)
            with (oracle/'samples.jsonl').open('x') as stream:
                for task_id in LOCK['protocol']['task_ids']:
                    case=cases[task_id]
                    stream.write(json.dumps({'id':task_id,'content':case['prompt']+case['canonical_solution'],'finish_reason':'stop'})+'\n')
            canonical=grade(args.cache,oracle,out,'oracle')
            validation=canonical_check(canonical)
            (out/'oracle-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        generations={};grades={}
        jobs=[('native-a','native'),('lsa-a','candidate'),('lsa-b','candidate'),('native-b','native')]
        (out/'jobs.json').write_text(json.dumps({'jobs':jobs,'split':args.split,'nodes':w.NODES,'candidate_config':str(candidate)},indent=2)+'\n')
        for label,profile in jobs:
            assert w.owns(token) and not errors
            assert not any(h['spark-request.json'] or h['spark3-request.json'] for h in w.each(w.HOLD_CODE,{'action':'read'}).values()),'pending requester'
            config=candidate if profile=='candidate' else Path(w.REL)/(profile+'.json')
            profile_action(config,'start',out,'boot-'+label)
            if label=='lsa-a' and args.benchmark in ('longbench','ruler'):
                with (out/'quality-observer-output.txt').open('x') as stream:
                    subprocess.Popen([sys.executable,str(HERE.parent/'2026-11-05-answer-quality/watch_quality.py'),
                        '--window',str(out),'--controller-root',str(ROOT)],cwd=ROOT,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            command=[sys.executable,str(HERE/'measure.py'),'--profile',str(config),
                     '--benchmark',args.benchmark,'--cache',str(args.cache),'--split',args.split,
                     '--private',str(private/label),'--out',str(out/('answers-'+label+'.json'))]
            if args.benchmark=='ruler':command+=['--ruler-stage',args.ruler_stage]
            w.run(command,out,'benchmark-'+label)
            generations[label]=json.loads((out/('answers-'+label+'.json')).read_text())
            grades[label]=grade(args.cache,private/label,out,label) if args.benchmark=='code' else generations[label]
            if label.startswith('lsa-'):
                result=early_result(generations,grades,label,args.benchmark)
                if not result['passed']:
                    (out/('quality-reject-'+label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
                    raise RuntimeError('observed final-answer regression; restoring before further benchmarks')
            (out/'completed.json').write_text(json.dumps({'at':w.stamp(),'completed':list(generations)},indent=2)+'\n')
        result=quality_result(generations,grades,args.benchmark)
        (out/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
        assert result['passed'],'final repeated-boot quality gate failed'
    except Exception:
        (out/'error.txt').write_text(traceback.format_exc());raise
    finally:
        try:restore(token,out,'final-restore',finish)
        finally:finish()


if __name__=='__main__':main()
