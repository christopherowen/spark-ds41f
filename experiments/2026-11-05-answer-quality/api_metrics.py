"""Server counters collected between timed phases, never inside requests."""
import datetime
import hashlib
import math
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from spark_ds41f.bench import parse_metric_series

SPEC = ('vllm:spec_decode_num_drafts_total','vllm:spec_decode_num_draft_tokens_total',
        'vllm:spec_decode_num_accepted_tokens_total')
SUCCESS = 'vllm:request_success_total'
LATENCIES = ('vllm:request_prefill_time_seconds','vllm:request_decode_time_seconds',
             'vllm:e2e_request_latency_seconds','vllm:time_to_first_token_seconds')


def counters(raw):
    series = parse_metric_series(raw)
    wanted = set(SPEC+(SUCCESS,'vllm:generation_tokens_total','vllm:num_preemptions_total',
                      'vllm:num_requests_running','vllm:num_requests_waiting'))
    wanted.update(name+suffix for name in LATENCIES for suffix in ('_sum','_count'))
    result = {}
    for (name,labels),value in series.items():
        if name in wanted:
            if not math.isfinite(value):raise ValueError('nonfinite server metric')
            result[name] = result.get(name,0.)+value
        if name in ('vllm:spec_decode_num_accepted_tokens_per_pos_total',
                    'vllm:spec_decode_num_draft_tokens_per_pos_total'):
            position = dict(labels).get('position')
            if position is not None:
                key = name+'@'+position
                result[key] = result.get(key,0.)+value
    if not all(name in result for name in SPEC):
        raise ValueError('required speculation counters are missing')
    return result


def snapshot(url,path):
    with urllib.request.urlopen(url.rstrip('/')+'/metrics',timeout=10) as response:
        raw = response.read()
    values = counters(raw.decode())
    if values.get('vllm:num_requests_running',0) or values.get('vllm:num_requests_waiting',0):
        raise ValueError('server is not idle before the phase')
    if path.exists():raise ValueError('refuse to overwrite a metric snapshot')
    path.write_bytes(raw)
    return {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'sha256':hashlib.sha256(raw).hexdigest(),'values':values}


def delta(before,after,expected_requests):
    a,b = before['values'],after['values']
    changes = {name:b.get(name,0)-a.get(name,0) for name in a.keys()|b.keys()
               if name not in ('vllm:num_requests_running','vllm:num_requests_waiting')}
    if any(value < 0 for value in changes.values()):raise ValueError('server counters reset during the phase')
    if changes.get(SUCCESS,0) != expected_requests:raise ValueError('request coverage differs from the phase')
    if b.get('vllm:num_requests_running',0) or b.get('vllm:num_requests_waiting',0):
        raise ValueError('server is not idle after the phase')
    drafts,verified,accepted = [changes[name] for name in SPEC]
    if accepted > verified:raise ValueError('accepted drafts exceed verified drafts')
    means = {}
    for name in LATENCIES:
        count = changes.get(name+'_count',0)
        means[name] = changes.get(name+'_sum',0)/count if count else None
    return {'before':{k:before[k] for k in ('at','sha256')},
            'after':{k:after[k] for k in ('at','sha256')},'deltas':changes,
            'accepted_per_draft':accepted/drafts if drafts else None,
            'accepted_fraction':accepted/verified if verified else None,
            'mean_server_seconds':means,'scope':'aggregate phase counters; collection outside all request timers'}


def finish(url,before,expected_requests,path,timeout=20):
    deadline = time.monotonic()+timeout
    while True:
        with urllib.request.urlopen(url.rstrip('/')+'/metrics',timeout=10) as response:
            raw = response.read()
        values = counters(raw.decode())
        done = values.get(SUCCESS,0)-before['values'].get(SUCCESS,0)
        idle = not values.get('vllm:num_requests_running',0) and not values.get('vllm:num_requests_waiting',0)
        if done >= expected_requests and idle:break
        if time.monotonic() >= deadline:raise ValueError('server metrics did not cover the completed phase')
        time.sleep(.5)
    if path.exists():raise ValueError('refuse to overwrite a metric snapshot')
    path.write_bytes(raw)
    after = {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'sha256':hashlib.sha256(raw).hexdigest(),'values':values}
    return delta(before,after,expected_requests)
