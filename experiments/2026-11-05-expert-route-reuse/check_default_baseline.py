"""Record the locked source's actual default code for comparison with the fix."""
import hashlib
import json
from pathlib import Path
import tilelang
from tilelang import tvm
from vllm.models.deepseek_v4_1.tilelang import moe

target=tvm.target.Target({'kind':'cuda','arch':'sm_121'})
cases=[]
for inter in (576,768):
    for stage in ('gate_up','down'):
        plan=moe.MoeExpertsV41Plan(384,6,5120,inter)
        assert plan.prefill_swizzle_panel==3
        factory,args,kwargs=plan.kernels()[f'prefill_{stage}']
        func=factory.get_tir(*args,**kwargs)
        with target,tvm.transform.PassContext(opt_level=3,config=factory.pass_configs):
            source=tilelang.lower(func,target=target,enable_device_compile=False).kernel_source
        assert 'rasterization2DRow<' not in source
        cases.append({'intermediate':inter,'stage':stage,
                      'default_source_sha256':hashlib.sha256(source.encode()).hexdigest()})
print('RESULT_JSON\n'+json.dumps({'base':'1f33102547e4da5c64712390604695d1abd577c4',
                                'runtime_source_sha256':hashlib.sha256(Path(moe.__file__).read_bytes()).hexdigest(),
                                'native_device_compilation':False,'gpu_execution':False,
                                'default_plan_cases':cases},indent=2))
