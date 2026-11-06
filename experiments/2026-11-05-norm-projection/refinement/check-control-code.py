import hashlib, json, pathlib, re
import tilelang
from tilelang import tvm
from vllm.models.deepseek_v4_1.tilelang.front import attention_front_kernel as native
from front_variants import attention_front_kernel as copy
outputs=[]
for f,args in ((native,(1e-20,256,"int64")),(copy,(1e-20,256,"int64",128,0))):
 f.target="cuda -arch=sm_121"
 func=f.get_tir(*args)
 with tvm.target.Target({"kind":"cuda","arch":"sm_121"}),tvm.transform.PassContext(config=f.pass_configs):
  out=tilelang.lower(func,target={"kind":"cuda","arch":"sm_121"},enable_host_codegen=False,enable_device_compile=False)
 print(type(out).__name__,list(vars(out)))
 source=out.kernel_source
 outputs.append(source)
 print(len(source),hashlib.sha256(source.encode()).hexdigest())
print("exact_generated_cuda_equal",outputs[0]==outputs[1])
if outputs[0]!=outputs[1]:
 import difflib
 print("".join(difflib.unified_diff(outputs[0].splitlines(True),outputs[1].splitlines(True)))[:10000])
