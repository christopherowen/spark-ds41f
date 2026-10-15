import importlib.util
from pathlib import Path
import tilelang
from tilelang import tvm

base = Path("/work/testing/python/language")


def load(name):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mx8 = load("test_tilelang_language_mxf8f6f4_mma_block_scale")
mx4 = load("test_tilelang_language_mxfp4_mma_block_scale")
for arch in ("sm_120a", "sm_121a"):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    for name, func in (
        ("MXFP8", mx8._make_mxfp8_matmul_kernel(64, 64, 128)),
        ("FP8xFP4", mx8._make_mxfp8_matmul_kernel(64, 64, 128, b_fp4=True)),
        ("MXFP4", mx4._make_mxfp4_matmul_kernel(64, 64, 128)),
    ):
        with target:
            result = tilelang.lower(func, target=target, enable_device_compile=True)
        print(f"{arch} {name}: CUDA device compilation passed", flush=True)
