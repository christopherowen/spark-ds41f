"""Private-operand compact-mean feasibility; never imported by serving."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from metadata import decode
from benchmark_runtime import graph_of,sample
from reuse_score import prepare_queries,prepared_scores
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import summarize_blocks
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK,IndexerV41Plan,indexer_score_v41,write_index_keys


def coverage(selected,expected,width):
    mask=torch.zeros(len(selected),width,dtype=torch.bool,device='cuda')
    mask.scatter_(1,selected.long(),True)
    return mask.gather(1,(expected//8).long()).float().mean().item()


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--replay-receipt',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();assert not args.out.exists()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    replay=json.loads(args.replay_receipt.read_text());assert replay['passed']
    paths=sorted(args.inputs.glob('validation-a-rank-0-*.pt'));assert len(paths)==4
    result={'passed':False,'coarse_component_only':True,'request_performance_measured':False,
        'model_quality_tested':False,'replay_sha256':sha(args.replay_receipt),'cases':[],
        'scope':'separate MXFP4 mean cache and native32 scorer; compaction outside score timer; not a serving candidate'}
    torch.cuda.reset_peak_memory_stats()
    for path in paths:
        assert sha(path)==replay['files'][path.name]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        page,width=data['page_size'],data['width'];small=page//8;n=width//8
        assert small==32,'this predeclared tile probe covers page256 captures'
        pool=torch.zeros(len(data['pool']),page*100+128,dtype=torch.uint8,device='cuda')[:,:page*100]
        pool[:,:page*68].copy_(data['pool'])
        complete=[i for i,count in enumerate(data['layout']['written_prefixes']) if count==page]
        slots=torch.tensor(complete,dtype=torch.int64,device='cuda')*page+page-1
        summarize_blocks(page)(pool,pool.view(torch.bfloat16),slots)
        means=pool.view(torch.bfloat16)[:,page*68//2:page*100//2].view(len(pool),small,128)
        packed=torch.zeros(len(pool),small*68+128,dtype=torch.uint8,device='cuda')[:,:small*68]
        flat=means.contiguous().view(-1,128);all_slots=torch.arange(len(flat),dtype=torch.int64,device='cuda')
        write_index_keys(flat,packed,all_slots,small)
        quantized=decode(packed[:,:small*64].reshape(-1,small,64),
                         packed[:,small*64:small*68].reshape(-1,small,4)).bfloat16()
        quantization=sample(graph_of(lambda:write_index_keys(flat,packed,all_slots,small)))
        for rows in (1,8,64):
            assert len(data['q'])>=rows
            q,scales,weights=[data[k][:rows].cuda() for k in ('q','scales','weights')]
            table=torch.zeros(rows,width//page,dtype=torch.int32,device='cuda')
            count=min(table.shape[1],data['table'].shape[1]);table[:,:count].copy_(data['table'][:rows,:count])
            lengths,active=data['lengths'][:rows].cuda(),data['active'].cuda()
            full=lengths//page*small;active_small=active//page*small
            source=bool(data['candidate_blocks']);budget=4096 if source else 512
            assert int(full.min())>budget
            decoded=torch.empty(rows,32,128,dtype=torch.bfloat16,device='cuda')
            scores={k:torch.empty(rows,n,dtype=torch.bfloat16,device='cuda') for k in ('bf16','fp4','oracle')}
            counts=torch.empty(rows,dtype=torch.int32,device='cuda')
            unpack=prepare_queries(width,page,budget);bf16=prepared_scores(width,page,budget)
            compact=indexer_score_v41(n,small,block_P=32)
            def baseline():
                unpack(q.view(torch.uint8),scales.view(torch.uint8).reshape(rows,32,4),lengths,active,decoded)
                bf16(decoded,scales.view(torch.uint8).reshape(rows,32,4),weights,means,table,lengths,active,scores['bf16'],counts)
            def candidate():
                compact(q.view(torch.int8),scales.view(torch.uint32).reshape(rows,32),weights,
                    packed.view(torch.int8),packed.view(torch.uint32),table,full,active_small,scores['fp4'])
            baseline();candidate()
            bf16(decoded,scales.view(torch.uint8).reshape(rows,32,4),weights,quantized,table,lengths,active,scores['oracle'],counts)
            exact=True;differences=0
            for row,extent in enumerate(full.cpu().tolist()):
                a=scores['fp4'][row,:extent];b=scores['oracle'][row,:extent]
                assert bool(torch.isfinite(a).all())
                diff=int((a.view(torch.int16)!=b.view(torch.int16)).sum());differences+=diff;exact=exact and not diff
            native=IndexerV41Plan(width,page)
            expected,_=native.run(q,scales,weights,pool,table,lengths,active)
            selector=IndexerTopK(n,budget)
            selected={k:torch.empty(rows,budget,dtype=torch.int32,device='cuda') for k in ('bf16','fp4')}
            for name in selected:
                values=torch.empty(rows,budget,dtype=torch.bfloat16,device='cuda')
                scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in selector.scratch_specs(rows)]
                selector.run(scores[name],full,active_small,selected[name],values,scratch=scratch)
            graphs={'bf16':graph_of(baseline),'fp4':graph_of(candidate)}
            before=sample(graphs['bf16']);new=sample(graphs['fp4']);after=sample(graphs['bf16'])
            import statistics
            row={'file':path.name,'sha256':sha(path),'rows':rows,'source':source,'page':page,
                 'visible_range':[int(lengths.min()),int(lengths.max())],
                 'quantized_mean_oracle_bits_equal':exact,'score_bit_differences':differences,
                 'bf16_native_top512_coverage':coverage(selected['bf16'],expected,n),
                 'fp4_native_top512_coverage':coverage(selected['fp4'],expected,n),
                 'bf16_before':before,'fp4':new,'bf16_after':after,
                 'bf16_over_fp4':statistics.median(before['samples_ms']+after['samples_ms'])/new['median_ms']}
            result['cases'].append(row);args.out.write_text(json.dumps(result,indent=2)+'\n')
            print('CASE '+json.dumps(row),flush=True)
            del graphs,native,expected,selected,selector,scratch,decoded,scores,q,scales,weights,table,lengths,full,active_small
            torch.cuda.empty_cache()
        result.setdefault('construction',[]).append({'file':path.name,'separate_snapshot_quantization':quantization,
            'additional_packed_mean_cache_bytes':packed.untyped_storage().nbytes(),
            'scope':'all allocated mean slots; current BF16 construction still required; no amortized request claim'})
        del pool,means,packed,flat,all_slots,quantized,data
        torch.cuda.empty_cache()
    result.update(passed=all(r['quantized_mean_oracle_bits_equal'] for r in result['cases']),
        memory={'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    args.out.write_text(json.dumps(result,indent=2)+'\n');print('RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
