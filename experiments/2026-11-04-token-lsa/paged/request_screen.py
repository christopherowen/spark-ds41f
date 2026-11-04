"""Explicit raw-token qualification and separate cold/cached request timing.

Private operands/captures remain on the lab node. Public receipts contain only
request identities, counts, elapsed time and response digests, never token IDs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def post(url,path,payload):
    request = urllib.request.Request(url.rstrip('/')+path,data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=900) as response:
        return json.load(response)


def make_corpus(url,directory):
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    target = directory/'requests.jsonl'
    if target.exists():
        return
    with urllib.request.urlopen(url.rstrip('/')+'/v1/models',timeout=30) as response:
        model = json.load(response)['data'][0]['id']
    cases = []
    short = [
        ('arithmetic','What is 347 times 29? Explain the calculation.'),
        ('code','Write a Python function that merges two sorted lists without sorting again.'),
        ('prose','Explain why rainbows appear, in one short paragraph.'),
        ('json','Return only JSON containing the first five prime numbers under the key primes.'),
        ('unicode','Translate into English: Dobar dan. Kako ste danas?'),
        ('reasoning','A box has 3 red and 5 blue balls. Two are drawn without replacement. What is the probability both are blue?'),
    ]
    for name,text in short:
        ids = post(url,'/tokenize',{'model':model,'messages':[{'role':'user','content':text}],
                                  'add_generation_prompt':True})['tokens']
        cases.append({'id':name,'payload':{'model':model,'prompt':ids,'temperature':0,
                      'seed':0,'max_tokens':128,'cache_salt':'page-gate-'+name}})
    paragraph = ('The queue stores pending work in arrival order. Each task has a stable '
                 'identifier and a priority. A worker removes one task, records its start, '
                 'performs it, and records its result. Failed work may be retried with '
                 'a bounded delay. The registry protects updates with one lock.\n')
    text = ('Review this engineering notebook. The registry code word is amber-731.\n'+
            paragraph*8000+'\nState the registry code word, then write a Python '
            'implementation of a thread-safe retry queue with bounded attempts. Explain '
            'the invariants and show a short example.\n')
    ids = post(url,'/tokenize',{'model':model,'messages':[{'role':'user','content':text}],
                              'add_generation_prompt':True})['tokens']
    assert len(ids) > 262144
    for length in (32768,131072,262144):
        prompt = ids[:length-2048]+ids[-2048:]
        assert len(prompt) == length
        name = f'notebook-{length}'
        cases.append({'id':name,'payload':{'model':model,'prompt':prompt,'temperature':0,
                     'seed':0,'max_tokens':128,'cache_salt':'page-gate-'+name}})
    with target.open('x') as handle:
        for case in cases:
            handle.write(json.dumps(case)+'\n')
    # Source revision and tokenizer must be verified separately against the
    # pinned deployment; this identity is not inferred from generated text.
    identity = {'checkpoint':'v41-flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170',
                'tokenizer':'v41-flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170',
                'model':model,'corpus_sha256':digest(cases)}
    (directory/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    (directory/'manifest.json').write_text(json.dumps({'cases':[
        {'id':c['id'],'prompt_tokens':len(c['payload']['prompt']),
         'input_sha256':digest({k:v for k,v in c['payload'].items() if k != 'cache_salt'})}
        for c in cases],'construction':'fixed chat template tokenization, long notebook trimmed in its middle',
        'scope':'synthetic deterministic screen; not production-agent quality qualification'},indent=2)+'\n')


def timed(url,case,payload):
    # Encoding, identity hashing and response comparison are outside the timer.
    raw = json.dumps(payload).encode()
    identity = digest({k:v for k,v in payload.items() if k != 'cache_salt'})
    request = urllib.request.Request(url.rstrip('/')+'/v1/completions',data=raw,
                                     headers={'Content-Type':'application/json'})
    began = time.perf_counter()
    with urllib.request.urlopen(request,timeout=900) as response:
        body = response.read()
    elapsed = time.perf_counter()-began
    result = json.loads(body)
    return {'id':case['id'],'input_sha256':identity,'seconds':elapsed,
            'usage':result['usage'],'choices_sha256':digest([
                {k:c.get(k) for k in ('index','text','finish_reason','stop_reason')}
                for c in result['choices']]),'raw_token_capture':False}


def benchmark(url,directory,out):
    cases = {c['id']:c for c in map(json.loads,(directory/'requests.jsonl').read_text().splitlines())}
    result = {'scope':'cold and same-prompt prefix-cache request screen; all candidate maintenance inside requests',
              'generated_token_gate':'separate capture/compare; no raw token IDs requested in timing',
              'cases':[]}
    # Cold work is charged once per boot; repeats are explicitly cached requests.
    for length,concurrency in ((131072,1),(131072,4),(262144,1)):
        case = cases[f'notebook-{length}']
        payloads = [dict(case['payload'],cache_salt=uuid.uuid4().hex) for _ in range(concurrency)]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            cold = list(pool.map(lambda payload:timed(url,case,payload),payloads))
            warm = [list(pool.map(lambda payload:timed(url,case,payload),payloads)) for _ in range(3)]
        result['cases'].append({'length':length,'concurrency':concurrency,'cold':cold,'cached_repeats':warm})
        Path(out).write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['corpus','bench'])
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--private',required=True)
    parser.add_argument('--out')
    options = parser.parse_args()
    if options.command == 'corpus':
        make_corpus(options.url,Path(options.private))
    else:
        assert options.out
        benchmark(options.url,Path(options.private),options.out)


if __name__ == '__main__':
    main()
