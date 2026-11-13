import os, torch, importlib
os.environ.setdefault("VLLM_L2_PREFETCH_KERNELS", "tilelang")
from vllm.models.glm5next.nvidia import l2_prefetch as m
print("ENABLED", m.ENABLED, "GRID", m._GRID, "KERNELS", m._KERNELS, "PERSIST_L2", repr(m.PERSIST_L2), "CUTE_OK", getattr(m, "_CUTE_OK", None), "MAX_TOKENS", m._MAX_TOKENS)
d = torch.device("cuda", 0)
torch.cuda.init(); torch.zeros(1, device=d)
try:
    from cuda.bindings import driver as cu
    import cuda.bindings as cb
    print("cuda.bindings", getattr(cb, "__version__", "?"))
    err, dev = cu.cuDeviceGet(0)
    for name in ("CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE", "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"):
        a = getattr(cu.CUdevice_attribute, name, None)
        print(name, cu.cuDeviceGetAttribute(a, dev) if a is not None else "no attr")
    for name in ("CU_LIMIT_PERSISTING_L2_CACHE_SIZE", "CU_LIMIT_MAX_L2_FETCH_GRANULARITY"):
        a = getattr(cu.CUlimit, name, None)
        print(name, cu.cuCtxGetLimit(a) if a is not None else "no limit")
    print("driver version", cu.cuDriverGetVersion())
except Exception as e:
    print("cuda.bindings FAILED", type(e).__name__, e)
print("configure_persisting_l2 ->", m.configure_persisting_l2(d))
print("torch", torch.__version__, "cuda rt", torch.version.cuda)
