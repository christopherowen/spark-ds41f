"""Verify declared calls cover binding after the compiler cache is frozen."""
import json
import tilelang
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import summarize_blocks
from vllm.models.deepseek_v4_1.tilelang.query_reuse import QueryReusePlan

cases = [
    (16384, 128, False, 64),
    (65536, 128, False, 64),
    (524288, 128, False, 64),
    (1048576, 256, True, 512),
    (1048576, 256, True, 1024),
    (131072, 256, True, 1024),
]
calls = [prepare.call(summarize_blocks, 128), prepare.call(summarize_blocks, 256)]
for width, page, source, budget in cases:
    calls.extend(QueryReusePlan.calls(width, page, candidate_blocks=source, budget=budget))
tilelang.prepare(calls, num_workers=4)
prepare.freeze("optional LSA block8 query-reuse declared-call qualification")
for page in (128, 256):
    summarize_blocks(page)
for width, page, source, budget in cases:
    QueryReusePlan(width, page, candidate_blocks=source, budget=budget)
print(
    "RESULT "
    + json.dumps(
        {
            "passed": True,
            "performance_measured": False,
            "cases": len(cases),
            "bind_after_freeze": True,
            "calls": len(calls),
        }
    ),
    flush=True,
)
