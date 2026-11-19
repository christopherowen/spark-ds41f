"""Out-of-band fused candidate check and timing only."""
import argparse,runpy,sys
from pathlib import Path
from vllm.models.deepseek_v4_1.tilelang import compact_shortlist
from plan import Plan
p=argparse.ArgumentParser();p.add_argument('--action',choices=['check','timing'],required=True)
args,rest=p.parse_known_args()
compact_shortlist.CompactShortlistPlan=Plan
script='check_runtime.py' if args.action=='check' else 'benchmark_runtime.py'
sys.argv=[script,*rest]
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
