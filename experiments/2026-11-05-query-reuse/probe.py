"""Out-of-band once-per-query decode check and timing."""
import argparse,json,runpy,sys
from pathlib import Path
from vllm.models.deepseek_v4_1.tilelang import block_shortlist
from reuse_score import QueryReusePlan
parser=argparse.ArgumentParser()
parser.add_argument('--action',choices=['check','timing'],required=True)
args,remaining=parser.parse_known_args()
block_shortlist.BlockMeanShortlistPlan=QueryReusePlan
print('PROBE_CONFIG '+json.dumps({'decode_once_per_query':True,'admitted_query_workspace':True,'runtime_default_unchanged':True}),flush=True)
script='check_runtime.py' if args.action=='check' else 'benchmark_runtime.py'
sys.argv=[script,*remaining]
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
