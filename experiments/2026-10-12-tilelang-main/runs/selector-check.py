import importlib.util
from pathlib import Path
import tilelang
from tilelang import tvm
from tvm import tirx

base = Path("/work/testing/python/language")


def load(name):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mx8 = load("test_tilelang_language_mxf8f6f4_mma_block_scale")
mx4 = load("test_tilelang_language_mxfp4_mma_block_scale")
for name, func in (
    ("MXFP8", mx8._make_mxfp8_matmul_kernel(64, 64, 128)),
    ("MXFP4", mx4._make_mxfp4_matmul_kernel(64, 64, 128)),
):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})
    with target:
        result = tilelang.lower(func, target=target, enable_device_compile=False)
    selectors = []

    def visit(node):
        if (
            isinstance(node, tirx.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tl.ptx_mma_block_scale"
        ):
            selectors.append((str(node.args[17]), str(node.args[19])))

    for _, lowered in result.device_mod.functions.items():
        tirx.stmt_functor.post_order_visit(lowered.body, visit)
    assert selectors, "No block-scaled MMA calls found"
    print(name, sorted(set(selectors)), flush=True)
    assert any(pair != ("0", "0") for pair in selectors), (
        f"{name}: every K atom selects byte 0"
    )
