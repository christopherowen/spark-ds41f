"""One-owner source-query screen using the complete source/handoff oracle."""
import argparse
import json
import traceback

import torch
from clustered_check import inputs
from source_handoff import check as check_source
from vllm.models.deepseek_v4_1.tilelang.source_indexer import SourceClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.grouped_source_indexer import GroupedSourceClusteredIndexerPlan


def check(width, rows, mode, grouped, cuda_graph, repeats):
    args = list(inputs(width, [width-1], mode, unwritten_cache=True))
    generator = torch.Generator(device='cuda').manual_seed(width + rows + 51)
    for index in (0, 1, 2):
        args[index] = args[index].repeat(rows, *([1] * (args[index].ndim-1)))
    if mode == 'random':
        args[0].random_(0, 256, generator=generator)
        args[2].copy_((torch.randn((rows,32), device='cuda', generator=generator)/64).bfloat16())
    extents = [max(0, width-1-7*r) for r in range(rows)]
    extents[0] = 0
    extents[-1] = width-1
    args[5] = torch.tensor(extents, dtype=torch.int32, device='cuda')
    owner = args[4]
    args[4] = owner.repeat(rows,1)

    class OwnedPlan:
        # The fixture explicitly creates one owner's table. Never infer serving
        # ownership by comparing page rows or looking at an allocation pointer.
        def __init__(self, width, page_size):
            self.plan = GroupedSourceClusteredIndexerPlan(width, page_size)
        def run(self, *args):
            owned_args = list(args)
            owned_args[4] = owner
            return self.plan.run(*owned_args)

    result = check_source(width, extents, mode, cuda_graph, repeats, fixture=tuple(args),
                          candidate_factory=OwnedPlan if grouped else SourceClusteredIndexerPlan)
    result.update(rows=rows, grouped_metadata=grouped,
                  metadata_elements=0 if result['native_fallback'] else
                      2 * (1 if grouped else rows) * (width//64) * 128)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--single-row-only', action='store_true')
    options = parser.parse_args()
    result = {'schema_version':1, 'passed':False, 'cases':[]}
    try:
        cases = [(4096,1,'random'), (65536,1,'random'), (65536,8,'random'),
                 (65536,128,'random'), (65536,128,'separable'), (65536,256,'random')]
        if options.quick:
            cases = [(65536,8,'random'), (65536,8,'separable')]
        if options.single_row_only:
            cases = [(w,r,m) for w,r,m in cases if r == 1]
        if not cases:
            raise ValueError('no cases selected')
        for width, rows, mode in cases:
            case = check(width,rows,mode,options.grouped,not options.no_graph,options.repeats)
            result['cases'].append(case)
            print(json.dumps(case),flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
