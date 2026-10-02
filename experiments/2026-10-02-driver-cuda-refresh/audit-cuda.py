"""Read actual loaded CUDA library versions after a bounded BF16 matmul."""
import ctypes as C
import hashlib
import importlib.metadata as m
import json
from pathlib import Path
import subprocess
import torch
from triton.backends.nvidia.compiler import get_ptxas
x=torch.ones((128,128),dtype=torch.bfloat16,device='cuda')
y=x@x
torch.cuda.synchronize()
assert bool(torch.all(y==128))
paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if '/' in line and any(n in line for n in ('libcublas.so','libcublasLt.so','libcudart.so','libcuda.so'))})
versions={}
for name in ('nvidia-cublas','nvidia-cuda-runtime','nvidia-cuda-nvrtc','nvidia-nvjitlink','torch','triton','nvidia-cutlass-dsl'):
 versions[name]=m.version(name)
rt=C.CDLL(next(p for p in paths if '/libcudart.so' in p));rv=C.c_int();assert rt.cudaRuntimeGetVersion(C.byref(rv))==0
cu=C.CDLL('libcuda.so.1');dv=C.c_int();assert cu.cuDriverGetVersion(C.byref(dv))==0
blas=C.CDLL(next(p for p in paths if '/libcublas.so' in p));h=C.c_void_p();bv=C.c_int()
assert blas.cublasCreate_v2(C.byref(h))==0
assert blas.cublasGetVersion_v2(h,C.byref(bv))==0
assert blas.cublasDestroy_v2(h)==0
ptxas=get_ptxas(121)
print(json.dumps({'packages':versions,'torch_build_cuda':torch.version.cuda,'runtime_api':rv.value,
                  'driver_api':dv.value,'cublas_api':bv.value,'triton_ptxas_path':ptxas.path,
                  'ptxas_version':subprocess.check_output([ptxas.path,'--version'],text=True),
                  'loaded_libraries':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}},indent=2))
