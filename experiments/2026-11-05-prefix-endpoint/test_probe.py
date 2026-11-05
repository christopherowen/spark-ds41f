import io
import json
import unittest
from unittest.mock import patch
import probe


def events(*values):
    return io.BytesIO(b''.join(b'data: '+json.dumps(v).encode()+b'\n\n' for v in values)+b'data: [DONE]\n\n')


class ProbeTest(unittest.TestCase):
    def test_counts_streamed_tokens_and_usage(self):
        response=events({'choices':[{'text':'','token_ids':[]}]},
            {'choices':[{'text':'answer','token_ids':list(range(64)),
                         'logprobs':{'token_logprobs':[-1.0]*64}}]},
            {'choices':[],'usage':{'prompt_tokens':256,'prompt_tokens_details':{'cached_tokens':128}}})
        with patch.object(probe,'post',return_value=response):
            r=probe.request('http://unused','model',list(range(256)),'salt','test')
        self.assertNotIn('error',r)
        self.assertEqual(r['uncached_tokens'],128)
        self.assertEqual(len(r['token_ids']),64)
        self.assertGreaterEqual(r['ttft_s'],0)

    def test_missing_usage_is_a_retained_failure(self):
        with patch.object(probe,'post',return_value=events({'choices':[]})):
            r=probe.request('http://unused','model',[1,2],'salt','missing')
        self.assertIn('error',r)
        self.assertEqual(r['label'],'missing')
        self.assertIn('wall_s',r)

    def test_cancel_does_not_claim_cache_or_ttft_measurement(self):
        with patch.object(probe,'post',return_value=events()),patch.object(probe.time,'sleep'):
            r=probe.request('http://unused','model',[1,2],'salt','cancel',cancel=True)
        self.assertTrue(r['cancelled_by_client'])
        self.assertIsNone(r['ttft_s'])
        self.assertNotIn('cached_tokens',r)

    def test_prefix_identity_is_not_suffix_identity(self):
        self.assertEqual(probe.common_prefix([1,2,3],[1,2,4]),2)
        self.assertEqual(probe.common_prefix([1,2,3],[9,2,3]),0)
        self.assertEqual(probe.common_prefix([1,2],[1,2,3]),2)


if __name__=='__main__':
    unittest.main()
