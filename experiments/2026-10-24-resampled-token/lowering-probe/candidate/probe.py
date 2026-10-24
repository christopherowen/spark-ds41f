import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _insert_resampled_kernel
from vllm.v1.worker.gpu.spec_decode.tilelang_resampled import insert_kernel
n,b=1,127
s=torch.zeros((n,6),device="cuda",dtype=torch.int64)
c=torch.zeros(n,device="cuda",dtype=torch.int32)
a=torch.zeros((n,b),device="cuda",dtype=torch.int64)
v=torch.zeros((n,b),device="cuda")
cu=torch.tensor([0,6],device="cuda",dtype=torch.int32)
m=torch.zeros(6,device="cuda",dtype=torch.int64)
t=torch.zeros(1,device="cuda")
k=_insert_resampled_kernel[(n,)](s,6,c,a,b,v,b,b,cu,m,t,PADDED_RESAMPLE_NUM_BLOCKS=128)
print("ORIGINAL PTX BEGIN",flush=True)
print(k.asm["ptx"],flush=True)
print("ORIGINAL PTX END",flush=True)
ours=insert_kernel(b,"float32","int64",32,6,b,b)
print("CANDIDATE CUDA BEGIN",flush=True)
print(ours.get_kernel_source(),flush=True)
print("CANDIDATE CUDA END",flush=True)
