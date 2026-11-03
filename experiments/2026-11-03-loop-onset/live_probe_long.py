#!/usr/bin/env python3
"""Bounded diagnostic HTTP requests to an unchanged, idle serving cluster.

Claims the documented hold exclusively, records identity and every raw reply,
and releases only its own hold. No cluster sync/start/stop or image mutation.
This is a correctness investigation, not a performance or release measurement.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess
import uuid


REMOTE = r'''
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

data = json.load(sys.stdin)
home = Path.home()
hold = home / 'spark-hold.json'
def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()
def get(path):
    with urllib.request.urlopen('http://127.0.0.1:8000' + path, timeout=5) as response:
        return response.read().decode()
def busy():
    return [line for line in get('/metrics').splitlines()
            if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))
            and float(line.rsplit(' ', 1)[1]) != 0]
def own():
    state = json.loads(hold.read_text())
    assert state['holder'] == data['holder'], 'Hold belongs to another holder'
    return state

action = data['action']
if action == 'identity':
    x = json.loads(subprocess.check_output(['docker', 'inspect', 'spark-ds41f']))[0]
    config = x['Config']
    print(json.dumps({'image_id':x['Image'], 'image_name':config['Image'],
        'started':x['State']['StartedAt'], 'cmd':config['Cmd'],
        'labels':{k:v for k,v in (config.get('Labels') or {}).items()
                  if k.startswith('local.spark-ds41f.')},
        'mount_targets':[m['Destination'] for m in x['Mounts']]}))
elif action == 'acquire':
    assert not (home / 'spark3-hold.json').exists(), 'Legacy hold exists'
    assert not any((home / name).exists() for name in ['spark-request.json', 'spark3-request.json']), 'Cluster requested'
    assert not busy(), 'Service has traffic'
    state = {'holder':data['holder'], 'since':now(), 'heartbeat':now(),
        'expected_end':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=45)).isoformat(),
        'rule':'Bounded loop-cause diagnostic requests only; serving state remains unchanged. No sync/start/stop/restart.'}
    with os.fdopen(os.open(hold, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
        json.dump(state, out)
    print(json.dumps(state))
elif action == 'request':
    state = own()
    assert not any((home / name).exists() for name in ['spark-request.json', 'spark3-request.json']), 'Cluster requested'
    assert not busy(), 'Service has traffic'
    state['heartbeat'] = now()
    hold.write_text(json.dumps(state))
    body = data['body']
    assert body.get('max_tokens', 0) <= 20000
    assert data['path'] in ['/v1/chat/completions', '/v1/completions']
    request = urllib.request.Request('http://127.0.0.1:8000' + data['path'],
        json.dumps(body).encode(), {'Content-Type':'application/json'})
    start = now()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = {'status':response.status, 'response':json.load(response)}
    except urllib.error.HTTPError as error:
        result = {'status':error.code, 'error_body':error.read().decode()}
    state['heartbeat'] = now()
    hold.write_text(json.dumps(state))
    print(json.dumps({'started':start, 'ended':now(), **result}))
elif action == 'release':
    own()
    idle = not busy()
    hold.unlink()
    print(json.dumps({'released':True, 'service_idle':idle, 'time':now()}))
else:
    raise ValueError(action)
'''


def remote(host, data):
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host,
         shlex.join(['python3', '-c', REMOTE])], input=json.dumps(data),
        capture_output=True, text=True, timeout=960,
    )
    if result.returncode:
        raise RuntimeError(f'{host}: {result.stderr.strip()}')
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    holder = 'loop-onset-probe-' + uuid.uuid4().hex[:12]
    receipt = {'holder':holder, 'started':dt.datetime.now(dt.timezone.utc).isoformat(),
               'cases_file':str(args.cases), 'purpose':'correctness diagnosis; no performance comparison',
               'identities':{}, 'requests':[]}
    acquired = False
    try:
        for host in ['dgx1', 'dgx2', 'dgx3']:
            receipt['identities'][host] = remote(host, {'action':'identity'})
        assert len({x['image_id'] for x in receipt['identities'].values()}) == 1
        receipt['hold'] = remote('dgx1', {'action':'acquire', 'holder':holder})
        acquired = True
        for case in cases:
            body = dict(case['body'], cache_salt=uuid.uuid4().hex)
            record = {'name':case['name'], 'path':case['path'], 'body':body}
            receipt['requests'].append(record)
            record.update(remote('dgx1', {'action':'request', 'holder':holder,
                                         'path':case['path'], 'body':body}))
            (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            response = record.get('response', {})
            choices = response.get('choices', [{}])
            print(json.dumps({'name':case['name'], 'status':record['status'],
                'finish_reason':choices[0].get('finish_reason'),
                'usage':response.get('usage')}), flush=True)
            if record['status'] != 200:
                raise RuntimeError('Diagnostic request failed; not sending further requests')
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        if acquired:
            receipt['release'] = remote('dgx1', {'action':'release', 'holder':holder})
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
