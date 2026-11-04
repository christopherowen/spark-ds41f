"""Out-of-band large-row workspace and non-power-of-two budget gates."""
import json
import traceback

import check_pages
from check_direct import case as compact_case
from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import DirectPageShortlistPlan


class WidePlan(DirectPageShortlistPlan):
    def __init__(self,width,page,*,candidate_blocks=False):
        super().__init__(width,page,candidate_blocks=candidate_blocks,budget=1536,row_cap=256)


def main():
    result = {'passed':False,'compaction':[],'native_lifecycle':[]}
    try:
        for width,rows in [(65536,9),(1048576,1)]:
            result['compaction'].append(compact_case(width,128,1536,rows))
        check_pages.PageShortlistPlan = WidePlan
        for rows in (1,255,256,257):
            case = check_pages.case(128,False,rows,8192)
            case.update(row_cap=256,budget=1536,
                        scope='all-key coverage: every row output also matches native exactly')
            result['native_lifecycle'].append(case)
            print(json.dumps(case),flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
