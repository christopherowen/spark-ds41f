"""The serving gate must reject resets that never preempted a request."""
import io
import json
import unittest
from unittest.mock import patch

import probe


def response(request_id=None):
    events = [
        {'choices': [{'token_ids': [10, 11], 'logprobs': {'token_logprobs': [-1., -2.]}}]},
        {'choices': [{'token_ids': [12, 13], 'logprobs': {'token_logprobs': [-3., -4.]},
                      'finish_reason': 'length'}]},
        {'choices': [], 'usage': {'prompt_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 0}}},
    ]
    if request_id is not None:
        for event in events:
            event['id'] = 'cmpl-' + request_id
    return io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))


class ProbeTest(unittest.TestCase):
    def run_request(self, after, reset_result=None, minimum=1):
        client = probe.Client('http://unused', 'model', None)
        if reset_result is None:
            reset_result = {'success': True}
        metric = lambda n: f'vllm:num_preemptions_total{{model_name="test"}} {n}\n'
        with patch.object(probe, 'require_window'), patch.object(probe.time, 'sleep'):
            with patch.object(client, 'open', return_value=response()), \
                 patch.object(client, 'metrics', side_effect=[metric(4)] + [metric(after)] * 10), \
                 patch.object(client, 'json', return_value=reset_result) as reset:
                result = client.request([1, 2], 'salt', 'recovered', 4, reset_at=2,
                                        minimum_preemptions=minimum)
                reset.assert_called_once_with('/reset_prefix_cache?reset_running_requests=true', {})
        return result

    def test_reset_and_server_preemption_are_both_required(self):
        valid = self.run_request(5)
        self.assertNotIn('error', valid)
        self.assertEqual(valid['reset']['observed_output_tokens'], 2)
        self.assertEqual(valid['reset']['preemptions_delta'], 1)
        self.assertIn('error', self.run_request(4))
        self.assertIn('error', self.run_request(5, {'success': False}))

    def test_concurrent_reset_must_preempt_the_whole_group(self):
        self.assertIn('error', self.run_request(5, minimum=3))
        self.assertNotIn('error', self.run_request(7, minimum=3))

    def test_missing_counter_is_not_a_zero_observation(self):
        with self.assertRaisesRegex(RuntimeError, 'does not expose'):
            probe.preemptions('unrelated_metric 1\n')

    def test_equal_partial_outputs_and_different_scores_fail(self):
        good = {'label': 'a', 'token_ids': [1, 2], 'logprobs': [-1., -2.]}
        broken = {**good, 'label': 'b', 'error': 'connection closed'}
        self.assertFalse(probe.compare(good, broken, 'recovery')['passed'])
        altered = {**good, 'label': 'c', 'logprobs': [-1., -2.01]}
        comparison = probe.compare(good, altered, 'recovery')
        self.assertFalse(comparison['passed'])
        self.assertEqual(comparison['logprobs_first_difference'], 1)

    def run_cancellation(self, aborted_after=1, running_after=0, response_id=True):
        client = probe.Client('http://unused', 'model', None)
        def metrics(aborted, running=0):
            return (
                f'vllm:request_success_total{{finished_reason="abort"}} {aborted}\n'
                f'vllm:num_requests_running {running}\n'
                'vllm:num_requests_waiting 0\n')
        stream = response('recovery-salt-producer' if response_id else None)
        with patch.object(probe, 'require_window'), patch.object(probe.time, 'sleep'), \
             patch.object(client, 'open', return_value=stream), \
             patch.object(client, 'metrics', side_effect=[metrics(0, 1)] +
                          [metrics(aborted_after, running_after)] * 60), \
             patch.object(client, 'json', return_value={'status': 'aborted', 'aborted': 1}) as abort:
            record = client.request([1, 2], 'salt', 'producer', 4, cancel_at=2)
            if response_id:
                abort.assert_called_once_with('/abort_requests', {
                    'request_ids': ['cmpl-recovery-salt-producer-0']})
            else:
                abort.assert_not_called()
        return record

    def test_cancellation_needs_server_confirmation_and_idle(self):
        record = self.run_cancellation()
        self.assertNotIn('error', record)
        self.assertTrue(record['cancel']['confirmed'])
        self.assertEqual(record['token_ids'], [10, 11])
        self.assertEqual(record['logprobs'], [-1., -2.])
        self.assertIn('wall_s', record)
        self.assertIn('error', self.run_cancellation(aborted_after=0))
        self.assertIn('error', self.run_cancellation(running_after=1))

    def test_cancellation_never_sends_an_unknown_or_empty_request_id(self):
        record = self.run_cancellation(response_id=False)
        self.assertIn('Cannot identify', record['error'])

    def test_absent_or_invalid_cancellation_metrics_fail_closed(self):
        for metrics in ('', 'vllm:request_success_total{finished_reason="abort"} NaN\n',
                        'vllm:request_success_total{finished_reason="abort"} -1\n'):
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(RuntimeError, 'lacks valid cancellation metric'):
                    probe.cancellation_metrics(metrics)

    def test_admission_barrier_failure_is_retained(self):
        from unittest.mock import Mock
        barrier = Mock()
        barrier.wait.side_effect = RuntimeError('barrier failed')
        client = probe.Client('http://unused', 'model', None)
        with patch.object(client, 'open') as request:
            record = client.request([1, 2], 'salt', 'producer', 4, barrier=barrier)
        request.assert_not_called()
        self.assertIn('barrier failed', record['error'])


if __name__ == '__main__':
    unittest.main()
