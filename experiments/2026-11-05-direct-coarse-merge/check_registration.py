"""Verify declared calls cover binding after the compiler cache is frozen."""
import json
import tilelang
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.compact_shortlist import CompactShortlistPlan

cases=[(width,page,source) for page in (128,256)
       for width,source in [(65536,False),(16384,False),(1048576,True),(32768,True)]]
calls=[]
for width,page,source in cases:
    calls.extend(CompactShortlistPlan.calls(width,page,candidate_blocks=source))
tilelang.prepare(calls,num_workers=4)
prepare.freeze('direct coarse declared-call qualification')
for width,page,source in cases:
    CompactShortlistPlan(width,page,candidate_blocks=source)
print('RESULT '+json.dumps({'passed':True,'performance_measured':False,
    'cases':len(cases),'bind_after_freeze':True,'calls':len(calls)}),flush=True)
