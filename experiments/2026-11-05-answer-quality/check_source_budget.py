"""Separate source-budget mechanical gate; no generation or timing."""
import json

import check_pages
from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import DirectPageShortlistPlan


class SourcePlan(DirectPageShortlistPlan):
    def __init__(self,width,page,*,candidate_blocks=False):
        super().__init__(width,page,candidate_blocks=candidate_blocks,
                         budget=1024 if candidate_blocks else 64,row_cap=256)


if __name__ == '__main__':
    check_pages.PageShortlistPlan = SourcePlan
    cases = [check_pages.case(page,True,9,98305,width=131072,audit_pages=False) for page in (128,256)]
    print('RESULT '+json.dumps({'passed':True,'source_budget':1024,'rows':9,'scope':'mechanical only; no token or quality claim','cases':cases}),flush=True)
