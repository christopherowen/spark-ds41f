"""Synthetic initialization/geometry gate for diagnostic capture, not real data."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback

import torch
from clustered_check import inputs
from snapshot_layout import restore_layout
from replay_capture import replay_file
from vllm.models.deepseek_v4_1.tilelang import clustered_capture
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan


def check(extent, duplicate):
    width, page_size = 65536, 256
    args = inputs(width, [extent], 'random', unwritten_cache=True)
    # Keep actual planning wider than visibility and use an overallocated pool.
    active = extent + 513
    args[6].fill_(active)
    if duplicate:
        args[4][0, (extent - 1) // page_size] = args[4][0, 0]
    clustered_capture._captured.clear()
    with tempfile.TemporaryDirectory() as directory:
        os.environ['VLLM_DS41_CLUSTER_CAPTURE'] = directory
        if extent < 32769:
            os.environ['VLLM_DS41_CLUSTER_CAPTURE_MIN_KEYS'] = '8192'
        else:
            os.environ.pop('VLLM_DS41_CLUSTER_CAPTURE_MIN_KEYS', None)
        clustered_capture.maybe_capture(*args, width, page_size, True, {'prefix': 'synthetic.layer', 'layer_id': 20})
        assert not list(Path(directory).glob('*.pt')), 'unarmed capture ran'
        Path(directory, 'arm').touch()
        clustered_capture.maybe_capture(*args, width, page_size, True, {'prefix': 'synthetic.layer', 'layer_id': 20})
        paths = list(Path(directory).glob('*.pt'))
        assert len(paths) == 1
        data = torch.load(paths[0], weights_only=True, map_location='cpu')
        assert data['width'] == width and int(data['active'][0]) == active
        assert data['candidate_blocks'] and data['layout']['query_rows'] == 1
        assert data['capture_context'] == {'prefix': 'synthetic.layer', 'layer_id': 20}
        for physical, count in zip(range(len(data['pool'])), data['layout']['written_prefixes']):
            assert not data['pool'][physical, count*64:page_size*64].count_nonzero()
            assert not data['pool'][physical, page_size*64+count*4:].count_nonzero()
        restored, span = restore_layout(data)
        torch.testing.assert_close(restored[4], args[4][:, :(extent + 255)//256], atol=0, rtol=0)
        assert restored[3].stride(0) == args[3].stride(0)
        assert restored[3].data_ptr() % 256 == args[3].data_ptr() % 256
        plan = IndexerV41Plan(width, page_size, candidate_blocks=True)
        blocks = [torch.empty((1, CANDIDATE_BLOCKS), device='cuda', dtype=torch.int32) for _ in range(2)]
        counts = [torch.empty((1,), device='cuda', dtype=torch.int32) for _ in range(2)]
        left, scores = plan.run(*args, candidate_out=blocks[0], candidate_out_lengths=counts[0])
        right, recovered = plan.run(*restored, candidate_out=blocks[1], candidate_out_lengths=counts[1])
        for a, b in [(left, right), (blocks[0], blocks[1]), (counts[0], counts[1]),
                     (scores[:, :extent].view(torch.int16), recovered[:, :extent].view(torch.int16))]:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        clustered_capture.maybe_capture(*args, width, page_size, True, {'prefix': 'synthetic.layer', 'layer_id': 20})
        assert len(list(Path(directory).glob('*.pt'))) == 1, 'same pool captured twice'
        clustered_capture.maybe_capture(*args, width, page_size, False, {'prefix': 'synthetic.layer', 'layer_id': 20})
        assert len(list(Path(directory).glob('*.pt'))) == 2, 'source and dense budgets were conflated'
        replay_result = replay_file(paths[0], cuda_graph=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1')
        assert replay_result['identical'] and replay_result['source_interface']
        return {'synthetic': True, 'full_candidate_replay_identical': True, 'extent': extent, 'duplicate_page': duplicate,
                'planned_width': width, 'active': active, 'address_span_bytes': span,
                'row_stride_bytes': restored[3].stride(0),
                'physical_pages': len(data['pool']), 'identical': True,
                'visible_score_bits_identical': True, 'source_handoff_identical': True,
                'written_prefix_only': True,
                'output_sha256': hashlib.sha256(left.cpu().numpy().tobytes()).hexdigest()}


def main():
    result = {'schema_version': 1, 'passed': False, 'cases': []}
    try:
        for extent, duplicate in [(8193, False), (32769, True)]:
            case = check(extent, duplicate)
            result['cases'].append(case)
            print(json.dumps(case), flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
