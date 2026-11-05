"""Compile both benchmark copy kernels without opening a GPU device."""
import resource
import tilelang
import tvm
from benchmark_launch_chain import copy_kernel
target=tvm.target.Target({"kind":"cuda","arch":"sm_121"})
for grid in (0,64):
    with target:
        artifact=tilelang.lower(copy_kernel.get_tir(grid),target=target,enable_device_compile=True)
    print("copy kernel compiled",grid,flush=True)
print("maxrss_kib",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
