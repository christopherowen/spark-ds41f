"""Select exact direct coarse merging for an out-of-band process only."""
import argparse,runpy,sys
from pathlib import Path
from vllm.models.deepseek_v4_1.tilelang import compact_shortlist
from merge import DirectCoarseTopK

Original=compact_shortlist.CompactShortlistPlan


class Plan(Original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.source and self.block_budget==4096 and self.width//8>4096:
            self.coarse=DirectCoarseTopK(self.width//8)


p=argparse.ArgumentParser();p.add_argument('--action',choices=['check','timing'],required=True)
args,rest=p.parse_known_args()
compact_shortlist.CompactShortlistPlan=Plan
script='check_runtime.py' if args.action=='check' else 'benchmark_runtime.py'
sys.argv=[script,*rest]
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
