import unittest

from api_metrics import counters,delta


class ApiMetrics(unittest.TestCase):
    def test_all_engines_and_per_position_counters_are_preserved(self):
        raw = '''vllm:spec_decode_num_drafts_total{engine="0",model_name="a b"} 2
vllm:spec_decode_num_drafts_total{engine="1",model_name="a b"} 3
vllm:spec_decode_num_draft_tokens_total{engine="0"} 10
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 7
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 4
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 1
'''
        parsed = counters(raw)
        self.assertEqual(parsed['vllm:spec_decode_num_drafts_total'],5)
        self.assertEqual(parsed['vllm:spec_decode_num_accepted_tokens_per_pos_total@0'],5)

    def test_missing_counters_never_become_zero_acceptance(self):
        with self.assertRaises(ValueError):counters('vllm:request_success_total 1\n')

    def test_phase_acceptance_and_contamination(self):
        keys = ['vllm:spec_decode_num_drafts_total','vllm:spec_decode_num_draft_tokens_total',
                'vllm:spec_decode_num_accepted_tokens_total','vllm:request_success_total']
        before = {'at':'before','sha256':'a','values':dict.fromkeys(keys,0)}
        after = {'at':'after','sha256':'b','values':dict(zip(keys,[10,40,25,2]))}
        result = delta(before,after,2)
        self.assertEqual(result['accepted_per_draft'],2.5)
        self.assertEqual(result['accepted_fraction'],.625)
        with self.assertRaises(ValueError):delta(before,after,1)
        after['values'][keys[0]] = -1
        with self.assertRaises(ValueError):delta(before,after,2)


if __name__ == '__main__':unittest.main()
