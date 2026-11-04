"""Untimed correctness process. Never imported by the candidate query module."""
import json
import traceback
import torch
from fixture import files,setup
from metadata import decode
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS


def check(data,args,plan,native_scores,native_indices):
    actual = plan.run(*args)
    counts = plan.candidate_lengths.cpu().tolist()
    lengths = args[5].cpu().tolist()
    active = int(args[6][0])
    for row,count in enumerate(counts):
        extent = min(lengths[row],active)
        blocks = plan.candidates[row,: (count+7)//8].cpu().long()
        if extent:
            assert len(blocks) and bool((blocks[1:] > blocks[:-1]).all())
            assert int(blocks[-1]) == (extent-1)//8
            assert bool((blocks >= 0).all())
            assert count == len(blocks)*8-(8-extent%8)%8
        else:
            assert count == 0 and len(blocks) == 0
        assert count <= plan.capacity
        ids = actual[row].cpu()
        live = ids[ids >= 0]
        assert bool((live < extent).all())
        assert len(live) == len(live.unique())
        positions = torch.arange(count,device='cuda')
        original = plan.candidates[row,positions//8].long()*8+positions%8
        torch.testing.assert_close(plan.final_scratch[0][row,:count].view(torch.int16),
                                   native_scores[row,original].view(torch.int16),atol=0,rtol=0)
        if plan.source:
            source = plan.blocks[row].cpu().long()
            source = source[source >= 0]
            assert bool((source[1:] > source[:-1]).all())
            if extent:
                assert int(source[-1]) == (extent-1)//8
            assert int(plan.block_lengths[row]) == len(source)*8-((8-extent%8)%8 if extent else 0)
        if extent <= 512:
            torch.testing.assert_close(actual[row],native_indices[row],atol=0,rtol=0)
    return actual.clone()


def main():
    result = {'passed':False,'cases':[],
              'scope':'out-of-band structural, evaluated-score and graph tests; generated-token equivalence not yet tested'}
    try:
        for path in files():
            data,args,plan,construction,span = setup(path)
            reference = IndexerV41Plan(data['width'],data['page_size'])
            expected,scores = reference.run(*args)
            actual = check(data,args,plan,scores,expected)
            complete = int(data['lengths'][0])//512*8
            query = decode(args[0].view(torch.uint8),args[1].view(torch.uint8).reshape(1,32,4))[0].double()
            dots = (plan.centers[:complete].double() @ query.T).to(torch.bfloat16).float().clamp(min=0)
            terms = (dots*args[2][0].float()[None]).to(torch.bfloat16).float()
            total = torch.zeros(complete,dtype=torch.float32,device='cuda')
            for h in range(32):
                total += terms[:,h]
            reference_centroids = total.to(torch.bfloat16).float()
            actual_centroids = plan.scores[0,:complete].float()
            torch.testing.assert_close(actual_centroids,reference_centroids,rtol=.02,atol=.02)
            maximum_centroid_error = float((actual_centroids-reference_centroids).abs().max())
            count = int(plan.candidate_lengths[0])
            overlap = len(set(actual[0].cpu().tolist()) & set(expected[0].cpu().tolist()))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                plan.run(*args)
            graph.replay()
            torch.testing.assert_close(plan.out,actual,atol=0,rtol=0)
            original_lengths,original_active = args[5].clone(),args[6].clone()
            for value in (0,37,511,512,513,int(data['lengths'][0])//2,int(data['lengths'][0])):
                args[5].fill_(value)
                expected,scores = reference.run(*args)
                graph.replay()
                eager = check(data,args,plan,scores,expected)
                graph.replay()
                torch.testing.assert_close(plan.out,eager,atol=0,rtol=0)
            args[5].copy_(original_lengths);args[6].copy_(original_active)
            if plan.source:
                plan.run(*args)
                reindex = IndexerV41Plan(CANDIDATE_BLOCKS*8,data['page_size'],candidates=True)
                next_out,_ = reindex.run(*args,plan.blocks,plan.block_lengths)
                assert bool(((next_out < args[5][:,None]) | (next_out == -1)).all())
            case = {'snapshot':path.name,'keys_scored':count,'native_top512_overlap':overlap,
                    'overlap_is_diagnostic_not_acceptance':True,'native_evaluated_score_bits_match':True,
                    'centroid_reference_max_abs_error':maximum_centroid_error,
                    'source_handoff_legal':bool(plan.source),'graph_visibility_passed':True,
                    'token_id_gate':'not run; snapshot operands cannot establish generated token equivalence',
                    'construction':construction}
            result['cases'].append(case);print(json.dumps(case),flush=True)
            del graph,plan,args
            torch.cuda.empty_cache()
        result['passed'] = len(result['cases']) == 4
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
