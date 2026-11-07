"""Final-answer-only RULER grading; no references or comparisons in serving."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'2026-11-05-answer-quality'))
from api_metrics import snapshot,finish
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def load_inputs(cache,split,stage='full'):
    reference=json.loads((HERE/(split+'.set.json')).read_text())
    manifest=json.loads((cache/'evaluation.json').read_text())
    assert sha(cache/'evaluation.json')==reference['evaluation_sha256']
    assert sha(cache/'requests.jsonl')==reference['requests_sha256']==manifest['requests_sha256']
    assert sha(cache/'grader.py')==reference['grader_sha256']==manifest['grader_sha256']
    cases=[json.loads(line) for line in (cache/'requests.jsonl').read_text().splitlines()]
    assert len(cases)==reference['case_count'] and len({c['id'] for c in cases})==len(cases)
    identity=reference['evaluation_sha256']
    if stage=='screen':
        assert split=='pilot','the lean screen uses only the separately frozen pilot subset'
        path=HERE/'screen.set.json';screen=json.loads(path.read_text())
        assert screen['parent_evaluation_sha256']==identity
        assert screen['parent_requests_sha256']==reference['requests_sha256']
        assert screen['grader_sha256']==reference['grader_sha256']
        cases=[c for c in cases if c['id'] in screen['ids']]
        assert [c['id'] for c in cases]==screen['ids'] and len(cases)==screen['case_count']==26
        identity=sha(path)
    else:assert stage=='full'
    return reference,manifest,cases,identity


def one(url,model,case,protocol,metric,private):
    payload={'model':model,'messages':[{'role':'user','content':case['prompt']}],
             'temperature':protocol['temperature'],'seed':protocol['seed'],'max_tokens':protocol['max_tokens'],
             'chat_template_kwargs':{'thinking':protocol['thinking'],'reasoning_effort':protocol['reasoning_effort']},
             'cache_salt':uuid.uuid4().hex}
    raw=json.dumps(payload).encode();request=urllib.request.Request(url.rstrip('/')+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'})
    started=time.perf_counter()
    with urllib.request.urlopen(request,timeout=protocol['request_timeout_s']) as response:body=response.read()
    ended=time.perf_counter();result=json.loads(body)
    assert result['model']==model
    choice=result['choices'][0];message=choice['message'];content=message.get('content') or ''
    assert isinstance(content,str)
    complete=choice['finish_reason']=='stop' and bool(content.strip())
    score=metric([content],[case['outputs']]) if complete else 0
    with (private/(case['id']+'.json')).open('x') as stream:json.dump({'response':result,'request_sha256':hashlib.sha256(raw).hexdigest()},stream)
    return {'id':case['id'],'task':case['task'],'family':case['family'],'generator_length':case['generator_length'],
            'prompt_sha256':case['prompt_sha256'],'prompt_tokens_untemplated':case['prompt_tokens_untemplated'],
            'seconds':ended-started,'started':started,'finished':ended,'score':score,'correct':complete and score==100,
            'status':'complete' if complete else 'incomplete','finish_reason':choice['finish_reason'],
            'usage':result['usage'],'final_sha256':hashlib.sha256(content.encode()).hexdigest(),
            'reasoning_characters':len(message.get('reasoning') or message.get('reasoning_content') or '')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--split',choices=['pilot','heldout'],required=True)
    p.add_argument('--stage',choices=['full','screen'],default='full')
    p.add_argument('--private',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8001')
    p.add_argument('--model',default='deepseek-v4.1-flash')
    args=p.parse_args()
    reference,manifest,cases,identity=load_inputs(args.cache,args.split,args.stage)
    spec=importlib.util.spec_from_file_location('pinned_ruler_grader',args.cache/'grader.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    assert not args.out.exists();args.private.mkdir(mode=0o700,parents=True,exist_ok=False)
    report={'benchmark':'RULER project protocol','complete':False,'split':args.split,'stage':args.stage,'manifest_sha256':identity,
            'full_manifest_sha256':reference['evaluation_sha256'],
            'protocol':manifest['protocol'],'records':[],'scope':manifest['scope']}
    before=snapshot(args.url,args.out.with_name(args.out.stem+'-metrics-before.txt'))
    for case in cases:
        metric=official.TASKS[case['family']]['metric_fn']
        row=one(args.url,args.model,case,manifest['protocol'],metric,args.private)
        assert row['usage']['prompt_tokens']+manifest['protocol']['max_tokens']<=1048576
        report['records'].append(row);args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ['id','score','status','seconds']}),flush=True)
    rows=report['records'];report['wall_seconds']=max(r['finished'] for r in rows)-min(r['started'] for r in rows)
    report['correct']=sum(r['correct'] for r in rows);report['mean_official_score']=sum(r['score'] for r in rows)/len(rows)
    report['server_metrics']=finish(args.url,before,len(rows),args.out.with_name(args.out.stem+'-metrics-after.txt'))
    report['complete']=True;args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
