"""One frozen HumanEvalPlus attempt per task; no execution or grading here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from api_metrics import snapshot,finish
from code_quality import LOCK,inputs


def one(url,model,case,private):
    spec = LOCK['protocol']
    payload = {'model':model,'messages':[{'role':'user','content':case['prompt']}],
               'temperature':spec['temperature'],'seed':spec['seed'],'max_tokens':spec['max_tokens'],
               'chat_template_kwargs':{'thinking':spec['thinking'],'reasoning_effort':spec['reasoning_effort']},
               'cache_salt':uuid.uuid4().hex}
    raw = json.dumps(payload).encode()
    identity = hashlib.sha256(raw).hexdigest()
    request = urllib.request.Request(url.rstrip('/')+'/v1/chat/completions',data=raw,
                                     headers={'Content-Type':'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request,timeout=spec['request_timeout_s']) as response:
        body = response.read()
    ended = time.perf_counter()
    result = json.loads(body)
    assert result['model'] == model
    choice = result['choices'][0];message = choice['message']
    content = message.get('content') or ''
    assert isinstance(content,str)
    complete = choice['finish_reason'] == 'stop' and bool(content.strip())
    task = case['id'].replace('/','-')
    with (private/(task+'.json')).open('x') as stream:json.dump({'request_sha256':identity,'response':result},stream)
    public = {'id':case['id'],'prompt_sha256':case['prompt_sha256'],'started':started,'finished':ended,
              'seconds':ended-started,'status':'complete' if complete else 'incomplete',
              'finish_reason':choice['finish_reason'],'usage':result['usage'],
              'final_sha256':hashlib.sha256(content.encode()).hexdigest(),'request_sha256':identity,
              'reasoning_characters':len(message.get('reasoning') or message.get('reasoning_content') or '')}
    sample = {'id':case['id'],'content':content,'finish_reason':choice['finish_reason']}
    return public,sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--model',default='deepseek-v4.1-flash')
    args = parser.parse_args()
    assert args.concurrency > 0 and not args.out.exists()
    args.private.mkdir(mode=0o700,parents=True,exist_ok=False)
    data = inputs(args.cache)
    path = args.cache/'requests.jsonl'
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    assert [c['id'] for c in cases] == LOCK['protocol']['task_ids']
    for c in cases:
        expected = LOCK['protocol']['instruction_prefix']+'\n```python\n'+data[c['id']]['prompt'].strip()+'\n```'
        assert c['prompt'] == expected and c['prompt_sha256'] == hashlib.sha256(expected.encode()).hexdigest()
    report = {'benchmark':'HumanEvalPlus','source_lock':LOCK,'concurrency':args.concurrency,
              'requests_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':[],
              'scope':'generation only; executable final-program grading is a separate restricted CPU job'}
    before = snapshot(args.url,args.out.with_name(args.out.stem+'-metrics-before.txt'))
    with (args.private/'samples.jsonl').open('x') as samples,ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one,args.url,args.model,c,args.private) for c in cases]
        for future in futures:
            public,sample = future.result()
            samples.write(json.dumps(sample)+'\n');samples.flush()
            report['records'].append(public)
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:public[k] for k in ['id','seconds','status']}),flush=True)
    rows = report['records']
    report['wall_seconds'] = max(r['finished'] for r in rows)-min(r['started'] for r in rows)
    report['server_metrics'] = finish(args.url,before,len(rows),args.out.with_name(args.out.stem+'-metrics-after.txt'))
    report['complete'] = True
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':main()
