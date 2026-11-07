"""Score completed benchmark answers, with all generation charged to wall time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
import uuid

from prepare import LOCK,digest,grader
from api_metrics import snapshot,finish


def one(url,case,payload,extract,private):
    # Only request send/read is timed. Grade final content after the response;
    # neither benchmark references nor comparisons enter the serving process.
    raw = json.dumps(payload).encode()
    identity = hashlib.sha256(raw).hexdigest()
    request = urllib.request.Request(url.rstrip('/')+'/v1/chat/completions',data=raw,
                                     headers={'Content-Type':'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=LOCK['protocol']['request_timeout_s']) as response:
            body = response.read()
        finished = time.perf_counter()
        result = json.loads(body)
    except (urllib.error.URLError,TimeoutError) as error:
        finished = time.perf_counter()
        return {'id':case['id'],'domain':case['domain'],'difficulty':case['difficulty'],
                'prompt_tokens_untemplated':case['prompt_tokens_untemplated'],
                'prompt_sha256':case['prompt_sha256'],'seconds':finished-started,
                'started':started,'finished':finished,'correct':False,'status':'request-error',
                'error_type':type(error).__name__,'expected':case['answer'],'predicted':None}
    assert result.get('model') == payload['model'], 'served model identity differs'
    choice = result['choices'][0]
    message = choice['message']
    content = message.get('content') or ''
    assert isinstance(content,str), 'unexpected final-content schema'
    predicted = extract(content)
    complete = choice['finish_reason'] == 'stop' and bool(content.strip())
    status = ('truncated' if choice['finish_reason'] == 'length' else
              'missing-final' if not content.strip() else
              'incomplete' if not complete else
              'unparsed-final' if predicted is None else 'complete')
    receipt = {'id':case['id'],'domain':case['domain'],'difficulty':case['difficulty'],
               'prompt_tokens_untemplated':case['prompt_tokens_untemplated'],
               'prompt_sha256':case['prompt_sha256'],'seconds':finished-started,
               'started':started,'finished':finished,'correct':complete and predicted == case['answer'],
               'status':status,'finish_reason':choice['finish_reason'],'expected':case['answer'],
               'predicted':predicted,'usage':result['usage'],'final_sha256':hashlib.sha256(content.encode()).hexdigest(),
               'request_sha256':identity,'reasoning_characters':len(message.get('reasoning') or message.get('reasoning_content') or '')}
    with (private/(case['id']+'.json')).open('x') as handle:
        json.dump({'request_sha256':identity,'response':result},handle)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--split',choices=['pilot','heldout'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--model',default='deepseek-v4.1-flash')
    args = parser.parse_args()
    assert not args.out.exists()
    args.private.mkdir(mode=0o700,parents=True,exist_ok=False)
    assert digest(args.cache/'selection.json') == LOCK['selection_sha256']
    selection = json.loads((args.cache/'selection.json').read_text())
    assert selection['protocol'] == LOCK['protocol'] and selection['dataset_sha256'] == LOCK['dataset']['sha256']
    path = args.cache/(args.split+'-requests.jsonl')
    assert digest(path) == selection['artifact_sha256'][path.name]
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    assert [c['id'] for c in cases] == selection[args.split]
    extract = grader(args.cache)
    spec = LOCK['protocol']
    payloads = [{'model':args.model,'messages':[{'role':'user','content':c['prompt']}],
                 'temperature':spec['temperature'],'seed':spec['seed'],'max_tokens':spec['max_tokens'],
                 'chat_template_kwargs':{'thinking':spec['thinking'],'reasoning_effort':spec['reasoning_effort']},
                 'cache_salt':uuid.uuid4().hex} for c in cases]
    report = {'split':args.split,'benchmark':'LongBench v2','selection_sha256':digest(args.cache/'selection.json'),
              'dataset_sha256':LOCK['dataset']['sha256'],'protocol':spec,'phases':[],
              'scope':'completed final answers against official ground truth; native output is not an answer key'}
    for phase in spec[args.split+'_phases']:
        private = args.private/phase['name'];private.mkdir(mode=0o700)
        if not phase['cache_reuse']:
            for payload in payloads:
                payload['cache_salt'] = uuid.uuid4().hex
        current = dict(phase,records=[])
        report['phases'].append(current)
        metric_prefix = args.out.with_name(args.out.stem+'-'+phase['name'])
        before = snapshot(args.url,metric_prefix.with_name(metric_prefix.name+'-metrics-before.txt'))
        # map preserves frozen task order even when completions finish out of order.
        with ThreadPoolExecutor(max_workers=phase['concurrency']) as pool:
            futures = [pool.submit(one,args.url,c,p,extract,private) for c,p in zip(cases,payloads)]
            for future in futures:
                row = future.result();current['records'].append(row)
                args.out.write_text(json.dumps(report,indent=2)+'\n')
                if row['status'] == 'request-error':
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError('service request error; partial receipt retained')
                print(json.dumps({'phase':phase['name'],'id':row['id'],'correct':row['correct'],
                                  'status':row['status'],'seconds':row['seconds']}),flush=True)
        rows = current['records']
        wall = max(r['finished'] for r in rows)-min(r['started'] for r in rows)
        correct = sum(r['correct'] for r in rows)
        current.update(wall_seconds=wall,correct=correct,total=len(rows),accuracy=correct/len(rows),
                       correct_answers_per_second=correct/wall,
                       median_request_seconds=statistics.median(r['seconds'] for r in rows))
        current['server_metrics'] = finish(args.url,before,len(rows),
            metric_prefix.with_name(metric_prefix.name+'-metrics-after.txt'))
        args.out.write_text(json.dumps(report,indent=2)+'\n')
    report['complete'] = True
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
