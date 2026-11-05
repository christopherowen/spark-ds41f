"""Out-of-band qualification of the existing larger dense budget."""
import argparse,runpy,sys
from pathlib import Path
from vllm.models.deepseek_v4_1.tilelang import compact_shortlist
Parent=compact_shortlist.CompactShortlistPlan
class Plan(Parent):
    def __init__(self,*args,**kwargs):
        if not kwargs.get('candidate_blocks',False):kwargs['budget']=128
        super().__init__(*args,**kwargs)
p=argparse.ArgumentParser();p.add_argument('--action',choices=['check','timing'],required=True)
a,rest=p.parse_known_args();compact_shortlist.CompactShortlistPlan=Plan
script='benchmark_runtime.py' if a.action=='timing' else 'check_runtime.py'
sys.argv=[script,*rest];runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
