#!/usr/bin/env python3
"""Compare uninterrupted and preempted continuations in an owned serving window.

Use guarded_probe.py so identity, cooling, fans, memory and thermal checks apply.
The dev-mode reset preempts active requests and discards their private windows.
The receipt must show both a successful reset and increased server preemptions.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import threading
import time
import urllib.request
from pathlib import Path

from window_lease import require_window


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preemptions(text):
    values = re.findall(r'^vllm:num_preemptions_total(?:\{[^\n]*\})?\s+(\S+)', text, re.M)
    if not values:
        raise RuntimeError('Server does not expose its preemption counter')
    return sum(float(value) for value in values)


def cancellation_metrics(text):
    def total(name, label=None):
        rows = re.findall(r'^' + re.escape(name) + r'(\{[^\n]*\})?\s+(\S+)', text, re.M)
        values = [float(value) for labels, value in rows if label is None or label in labels]
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            raise RuntimeError(f'Server lacks valid cancellation metric {name}')
        return sum(values)
    return {
        'aborted': total('vllm:request_success_total', 'finished_reason="abort"'),
        'running': total('vllm:num_requests_running'),
        'waiting': total('vllm:num_requests_waiting'),
    }


class Client:
    def __init__(self, url, model, window_state):
        self.url, self.model, self.window_state = url.rstrip('/'), model, window_state

    def open(self, path, body=None, request_id=None):
        require_window(self.window_state)
        headers = {'Content-Type': 'application/json'}
        if request_id:
            headers['X-Request-Id'] = request_id
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(
            urllib.request.Request(self.url + path, data=data, headers=headers), timeout=900
        )

    def json(self, path, body=None):
        with self.open(path, body) as response:
            return json.load(response)

    def metrics(self):
        with self.open('/metrics') as response:
            return response.read().decode()

    def request(self, prompt, salt, label, tokens, reset_at=None, barrier=None,
                minimum_preemptions=1, cancel_at=None):
        record = {
            'label': label, 'prompt_tokens': len(prompt), 'prompt_sha256': digest(prompt),
            'cache_salt': salt, 'request_id': f'recovery-{salt}-{label}',
            'token_ids': [], 'logprobs': [], 'token_times_s': [], 'reset_at': reset_at,
            'cancel_at': cancel_at,
        }
        started = time.monotonic()
        try:
            if cancel_at is not None and (reset_at is not None or not 0 < cancel_at < tokens):
                raise ValueError('Cancellation needs an intermediate threshold and no reset')
            if barrier is not None:
                barrier.wait(timeout=60)
            with self.open('/v1/completions', {
                'model': self.model, 'prompt': prompt, 'cache_salt': salt,
                'temperature': 0, 'max_tokens': tokens, 'ignore_eos': True,
                'logprobs': 1, 'return_token_ids': True, 'stream': True,
                'stream_options': {'include_usage': True},
            }, record['request_id']) as response:
                for line in response:
                    require_window(self.window_state)
                    if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    if event.get('id'):
                        if record.get('response_id', event['id']) != event['id']:
                            raise RuntimeError('Completion response identity changed')
                        record['response_id'] = event['id']
                    if event.get('usage'):
                        record['usage'] = event['usage']
                    for choice in event.get('choices', []):
                        ids = choice.get('token_ids') or []
                        record['token_ids'].extend(ids)
                        record['token_times_s'].extend([time.monotonic() - started] * len(ids))
                        record['logprobs'].extend((choice.get('logprobs') or {}).get('token_logprobs') or [])
                        if choice.get('finish_reason'):
                            record['finish_reason'] = choice['finish_reason']
                    if reset_at is not None and len(record['token_ids']) >= reset_at and 'reset' not in record:
                        if len(record['token_ids']) >= tokens:
                            raise RuntimeError('Request finished before reset could interrupt it')
                        before = self.metrics()
                        reset_start = time.monotonic()
                        result = self.json('/reset_prefix_cache?reset_running_requests=true', {})
                        record['reset'] = {
                            'observed_output_tokens': len(record['token_ids']),
                            'start_s': reset_start - started, 'duration_s': time.monotonic() - reset_start,
                            'response': result, 'metrics_before': before,
                        }
                        if result != {'success': True}:
                            raise RuntimeError(f'Reset did not succeed: {result!r}')
                    if cancel_at is not None and len(record['token_ids']) >= cancel_at:
                        if len(record['token_ids']) >= tokens or record.get('finish_reason'):
                            raise RuntimeError('Producer finished before cancellation')
                        response_id = record.get('response_id')
                        if response_id != 'cmpl-' + record['request_id']:
                            raise RuntimeError('Cannot identify the exact producer to cancel')
                        before = self.metrics()
                        cancellation_metrics(before)
                        # Completion serving appends the prompt index to its
                        # response id. This workload always has exactly one prompt.
                        engine_id = response_id + '-0'
                        result = self.json('/abort_requests', {'request_ids': [engine_id]})
                        record['cancel'] = {
                            'observed_output_tokens': len(record['token_ids']),
                            'engine_request_id': engine_id, 'response': result,
                            'metrics_before': before,
                        }
                        if result != {'status': 'aborted', 'aborted': 1}:
                            raise RuntimeError(f'Cancellation was not acknowledged: {result!r}')
                        break
            if cancel_at is not None:
                if 'cancel' not in record:
                    raise RuntimeError('No cancellation occurred')
                before = cancellation_metrics(record['cancel']['metrics_before'])
                for _ in range(60):
                    after = self.metrics()
                    counts = cancellation_metrics(after)
                    delta = counts['aborted'] - before['aborted']
                    if delta >= 1 and counts['running'] == counts['waiting'] == 0:
                        break
                    time.sleep(0.5)
                record['cancel'].update(metrics_after=after, aborted_delta=delta,
                                        running=counts['running'], waiting=counts['waiting'])
                if delta < 1 or counts['running'] != 0 or counts['waiting'] != 0:
                    raise RuntimeError('Cancellation did not finish and return the server to idle')
                if len(record['token_ids']) != len(record['logprobs']) or not all(
                    value is not None and math.isfinite(value) for value in record['logprobs']
                ):
                    raise RuntimeError('Cancelled producer has incomplete or non-finite scores')
                record['cancel']['confirmed'] = True
                return record
            if reset_at is not None:
                if 'reset' not in record:
                    raise RuntimeError('No reset occurred')
                # Metrics can be exported on the next engine update.
                for _ in range(10):
                    after = self.metrics()
                    delta = preemptions(after) - preemptions(record['reset']['metrics_before'])
                    if delta > 0:
                        break
                    time.sleep(0.5)
                record['reset'].update(metrics_after=after, preemptions_delta=delta)
                if delta < minimum_preemptions:
                    raise RuntimeError(f'Reset preempted {delta} requests; expected at least {minimum_preemptions}')
            if len(record['token_ids']) != tokens or len(record['logprobs']) != tokens:
                raise RuntimeError('Incomplete token IDs or scores')
            if not all(value is not None and math.isfinite(value) for value in record['logprobs']):
                raise RuntimeError('Non-finite output score')
            if record.get('finish_reason') != 'length':
                raise RuntimeError(f"Unexpected finish: {record.get('finish_reason')}")
            usage = record['usage']
            if usage['prompt_tokens'] != len(prompt):
                raise RuntimeError('Server prompt count differs from tokenized input')
            record['cached_tokens'] = usage['prompt_tokens_details']['cached_tokens']
        except Exception as error:
            record['error'] = repr(error)
        finally:
            record['wall_s'] = time.monotonic() - started
            if record['token_times_s']:
                record['ttft_s'] = record['token_times_s'][0]
                record['output_tokens_per_s'] = len(record['token_ids']) / record['wall_s']
        return record


def compare(reference, actual, kind):
    result = {'reference': reference['label'], 'actual': actual['label'], 'kind': kind}
    result['complete'] = 'error' not in reference and 'error' not in actual
    for field in ('token_ids', 'logprobs'):
        left, right = reference[field], actual[field]
        result[field + '_equal'] = left == right
        result[field + '_first_difference'] = next(
            (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
            None if len(left) == len(right) else min(len(left), len(right)),
        )
    result['passed'] = result['complete'] and result['token_ids_equal'] and result['logprobs_equal']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--model', default='deepseek-v4.1-flash')
    parser.add_argument('--window-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recovery-policy', choices=['control', 'candidate'], required=True)
    parser.add_argument('--output-tokens', type=int, default=1536)
    parser.add_argument('--minimum-prompt-tokens', type=int, default=16384)
    parser.add_argument('--reset-at', type=int, nargs='+', default=[129, 257, 1025])
    args = parser.parse_args()
    if not args.reset_at or min(args.reset_at) < 1 or max(args.reset_at) >= args.output_tokens - 128:
        parser.error('Each reset must leave at least 128 output tokens for comparison')
    client = Client(args.url, args.model, args.window_state)
    require_window(args.window_state)
    args.out.mkdir(parents=True, exist_ok=False)
    nonce = str(time.time_ns())
    prompts, transcripts = [], []
    topics = [
        'Explain cancellation, ownership and safe recovery of a shared prefix cache.',
        'Review how independent cache groups can be evicted without reusing incomplete state.',
        'Describe tests for deterministic continuation after preemption and batch reordering.',
    ]
    source = Path(__file__).read_text()
    for topic in topics:
        messages = [
            {'role': 'system', 'content': 'Give a detailed technical review grounded in the supplied source.'},
            {'role': 'user', 'content': topic + '\n\n' + source},
        ]
        while True:
            ids = client.json('/tokenize', {'model': args.model, 'messages': messages})['tokens']
            if len(ids) >= args.minimum_prompt_tokens:
                break
            messages[-1]['content'] += '\nAdditional source listing:\n' + source
        prompts.append(ids)
        transcripts.append(messages)
    (args.out / 'workload.json').write_text(json.dumps({
        'transcripts': transcripts, 'prompts': prompts, 'hashes': [digest(p) for p in prompts],
        'output_tokens': args.output_tokens, 'reset_at': args.reset_at,
        'policy': args.recovery_policy,
    }, indent=2) + '\n')
    receipts, comparisons = [], []
    with (args.out / 'requests.jsonl').open('x') as output:
        def save(record):
            receipts.append(record)
            output.write(json.dumps(record) + '\n')
            output.flush()
            return record

        # Fresh salts make the uninterrupted controls cold; same-salt resends
        # establish ordinary prefix equivalence before forced recovery.
        baseline = []
        for i, prompt in enumerate(prompts):
            salt = f'{nonce}-prompt-{i}'
            baseline.append(save(client.request(prompt, salt, f'cold-{i}', args.output_tokens)))
            repeated = save(client.request(prompt, salt + '-repeat', f'cold-repeat-{i}', args.output_tokens))
            comparisons.append(compare(baseline[-1], repeated, 'baseline-stability'))
            warm = save(client.request(prompt, salt, f'warm-{i}', args.output_tokens))
            comparisons.append(compare(baseline[-1], warm, 'ordinary-prefix'))
            for threshold in args.reset_at:
                recovered = save(client.request(prompt, salt, f'recover-{i}-{threshold}',
                                                args.output_tokens, reset_at=threshold))
                comparisons.append(compare(baseline[-1], recovered, 'recovery'))
            # A new salt makes this producer the sole source of its prefix. Its
            # follower must reuse actual cached tokens after confirmed cancellation.
            cancel_salt = salt + '-cancelled-producer'
            cancelled = save(client.request(prompt, cancel_salt, f'cancelled-{i}',
                                            args.output_tokens, cancel_at=args.reset_at[0]))
            follower = save(client.request(prompt, cancel_salt, f'after-cancel-{i}',
                                           args.output_tokens))
            comparison = compare(baseline[-1], follower, 'cancelled-producer-prefix')
            comparison['cancellation_confirmed'] = (
                'error' not in cancelled and cancelled.get('cancel', {}).get('confirmed', False))
            comparison['prefix_reused'] = follower.get('cached_tokens', 0) > 0
            comparison['passed'] &= comparison['cancellation_confirmed'] and comparison['prefix_reused']
            comparisons.append(comparison)
        # Two admission orders, distinct traffic, and one global reset that
        # preempts every admitted request. Each prompt is compared with its
        # own solo baseline; only one stream initiates the reset.
        for order in ((0, 1, 2), (2, 0, 1)):
            barrier = threading.Barrier(len(order))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(order)) as pool:
                jobs = [(i, pool.submit(client.request, prompts[i], f'{nonce}-prompt-{i}',
                         f'concurrent-{order}-{i}', args.output_tokens,
                         args.reset_at[0] if j == 0 else None, barrier, len(order))) for j, i in enumerate(order)]
                for i, job in jobs:
                    comparisons.append(compare(baseline[i], save(job.result()), 'concurrent-recovery'))
    report = {
        'policy': args.recovery_policy, 'requests': len(receipts), 'comparisons': comparisons,
        'errors': [r for r in receipts if 'error' in r],
        'baseline_stable': all(c['passed'] for c in comparisons if c['kind'] == 'baseline-stability'),
        'passed': all(c['passed'] for c in comparisons) and not any('error' in r for r in receipts),
        'note': 'A control mismatch is an observed result, not a passing correctness gate. If uninterrupted A/A outputs differ, recovery comparisons are inconclusive until that numerical variability is resolved.',
    }
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'requests': report['requests'],
                      'failed_comparisons': sum(not c['passed'] for c in comparisons)}))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
