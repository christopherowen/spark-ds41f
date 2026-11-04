"""Performance process: no correctness reference or comparison in timed path."""
import argparse
import hashlib
import json
import statistics
import traceback
import torch
from fixture import files,setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['candidate','control'],required=True)
    parser.add_argument('--budget',type=int,default=64)
    parser.add_argument('--rows',type=int,default=1)
    parser.add_argument('--extend',type=int,default=0)
    options = parser.parse_args()
    result = {'passed':False,'mode':options.mode,'cases':[],
              'scope':'query-only snapshot kernel timing; construction separately reported; not request speed'}
    try:
        for path in files():
            torch.cuda.reset_peak_memory_stats()
            candidate = options.mode == 'candidate'
            data,args,plan,construction,span = setup(path,options.budget,options.rows,candidate,options.extend)
            if candidate:
                def run():
                    plan.run(*args)
            else:
                def run():
                    plan.run(*args,out=plan.out_buffer,scratch=plan.scratch_buffer,
                             **({'candidate_out':plan.blocks_buffer,'candidate_out_lengths':plan.counts_buffer}
                                if data['candidate_blocks'] else {}))
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            times = []
            for _ in range(20):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graph.replay()
                end.record();end.synchronize()
                times.append(start.elapsed_time(end)/50)
            case = {'snapshot':path.name,'extent':int(data['lengths'][0]),'rows':options.rows,
                    'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'source':bool(data['candidate_blocks']),'budget':512 if data['candidate_blocks'] else options.budget,
                    'planned_width':data['width'],'page_size':data['page_size'],'native_pool_span_bytes':span,
                    'synthetic_repeated_pages':bool(options.extend),
                    'query_ms_samples':times,'query_ms_median':statistics.median(times),
                    'construction':construction,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            if candidate:
                # Observation after all timers, never a reference/comparison.
                case['keys_scored'] = plan.candidate_lengths.cpu().tolist()
                case['construction_amortized_ms_per_query'] = {
                    str(n):construction['construction_wall_ms']/n for n in (1,8,256,1024)}
            result['cases'].append(case)
            print(json.dumps(case),flush=True)
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
