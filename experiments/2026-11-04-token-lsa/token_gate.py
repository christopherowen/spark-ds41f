"""Out-of-band raw generated-token capture/comparison. Never a benchmark client."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def choices(response):
    result = []
    for choice in sorted(response['choices'],key=lambda x:x['index']):
        ids = choice.get('token_ids')
        if not isinstance(ids,list) or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('raw generated token IDs missing; decoded text cannot substitute')
        result.append({key:choice.get(key) for key in ('index','token_ids','finish_reason','stop_reason')})
    if not result:
        raise ValueError('response has no choices')
    return result


def compare(reference,candidate):
    for key in ('checkpoint','tokenizer'):
        if reference['identity'][key] != candidate['identity'][key]:
            raise ValueError(key+' differs')
    base,actual = reference['records'],candidate['records']
    if not base or not actual or any(len({x['id'] for x in records}) != len(records) for records in (base,actual)):
        raise ValueError('nonempty unique request IDs required')
    for records in (base,actual):
        for record in records:
            choices({'choices':record['choices']})
    if [(x['id'],x['input_sha256']) for x in base] != [(x['id'],x['input_sha256']) for x in actual]:
        raise ValueError('workload identities/order differ')
    differences,matched = [],0
    for expected,observed in zip(base,actual):
        if expected['choices'] != observed['choices'] or expected['completion_tokens'] != observed['completion_tokens']:
            differences.append({'id':expected['id'],
                'reference_tokens_sha256':digest(expected['choices']),
                'candidate_tokens_sha256':digest(observed['choices'])})
        else:
            matched += 1
    return {'passed':not differences,'matched_requests':matched,'total_requests':len(base),
            'differences':differences,'criterion':'ordered raw generated IDs, count and termination',
            'scope':'declared deterministic corpus only; no universal equivalence or performance claim'}


def capture(options):
    identity = json.loads(Path(options.identity).read_text())
    assert identity.get('checkpoint') and identity.get('tokenizer')
    records = []
    for line in Path(options.requests).read_text().splitlines():
        case = json.loads(line)
        payload = dict(case['payload'])
        assert payload.get('temperature') == 0, 'token-equivalence corpus must declare deterministic greedy generation'
        payload.update(stream=False,return_token_ids=True)
        endpoint = case.get('endpoint','/v1/completions')
        assert endpoint in ('/v1/completions','/v1/chat/completions')
        request = urllib.request.Request(options.url.rstrip('/')+endpoint,
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        response = json.load(urllib.request.urlopen(request,timeout=options.timeout))
        records.append({'id':case['id'],'input_sha256':digest({'endpoint':endpoint,'payload':
            {k:v for k,v in payload.items() if k != 'cache_salt'}}),
            'prompt_token_ids':response.get('prompt_token_ids') or [x.get('prompt_token_ids') for x in response['choices']],
            'choices':choices(response),'completion_tokens':response['usage']['completion_tokens']})
    assert records and len({x['id'] for x in records}) == len(records)
    result = {'schema_version':1,'identity':identity,'records':records,
              'scope':'out-of-band deterministic raw-token capture; no timing'}
    with Path(options.out).open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    take = sub.add_parser('capture')
    for name in ('url','requests','identity','out'):
        take.add_argument('--'+name,required=True)
    take.add_argument('--timeout',type=int,default=900)
    check = sub.add_parser('compare')
    check.add_argument('reference');check.add_argument('candidate')
    options = parser.parse_args()
    if options.command == 'capture':
        capture(options)
        return 0
    result = compare(json.loads(Path(options.reference).read_text()),json.loads(Path(options.candidate).read_text()))
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
