"""Out-of-band load scheduling check/timing; serving defaults are unchanged."""
import argparse,json,runpy,sys
from pathlib import Path
from vllm.models.deepseek_v4_1.tilelang import block_shortlist
from load_score import block_scores
parser=argparse.ArgumentParser()
parser.add_argument('--action',choices=['check','timing'],required=True)
args,remaining=parser.parse_known_args()
block_shortlist.block_scores=block_scores
print('PROBE_CONFIG '+json.dumps({'single_shared_store':True,'runtime_default_unchanged':True}),flush=True)
script='check_runtime.py' if args.action=='check' else 'benchmark_runtime.py'
sys.argv=[script,*remaining]
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
