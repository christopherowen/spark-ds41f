"""CPU-only validation of the compiler pass that rejected the split prototype."""
import json
import sys
import tilelang
from tilelang import tvm
from front_variants import attention_front_kernel, attention_q_kernel, attention_kv_kernel

for factory, threads, partition in ((attention_front_kernel,64,0),(attention_front_kernel,128,0),(attention_front_kernel,256,0),(attention_q_kernel,128,1),(attention_kv_kernel,128,2)):
    factory.target = "cuda -arch=sm_121"
    function = factory.get_tir(1e-6,256,"int64",threads,partition)
    module = tvm.IRModule.from_expr(function)
    tilelang.transform.CanonicalizeLegacyReducer()(module)
    with tvm.target.Target({"kind":"cuda","arch":"sm_121"}), tvm.transform.PassContext(config=factory.pass_configs):
        tilelang.lower(function,target={"kind":"cuda","arch":"sm_121"},enable_host_codegen=False,enable_device_compile=False)
    print(json.dumps({"full_ir_lowering":True,"factory":factory.func.__name__,"threads":threads,"partition":partition,"legacy_reducer_pass":True}),flush=True)
print("CPU compiler-pass check only; no GPU kernel or performance measurement.",flush=True)
