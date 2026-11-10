"""Select an experimental coarse tile in an out-of-band process only."""
import argparse,runpy,sys
from pathlib import Path
from vllm.models.deepseek_v4_1.tilelang import compact_shortlist
from tile_score import indexer_score_v41

def score(width,page_size,**kwargs):
    kwargs['block_P']=64
    return indexer_score_v41(width,page_size,**kwargs)

parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['check','timing'],required=True)
args,rest=parser.parse_known_args()
compact_shortlist.indexer_score_v41=score
script='check_runtime.py' if args.action=='check' else 'benchmark_runtime.py'
sys.argv=[script,*rest]
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
