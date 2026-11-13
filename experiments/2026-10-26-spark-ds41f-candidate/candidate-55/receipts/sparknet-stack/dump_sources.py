import importlib.util, re
import sparknet
assert sparknet.__file__.startswith("/work/sparknet/"), sparknet.__file__
from sparknet.oneshot import _oneshot_tilelang as AR, _allgather_tilelang as AG
from sparknet.oneshot import _reducescatter_tilelang as RS, _exchange_tilelang as EX
spec = importlib.util.spec_from_file_location("pdl_sources", "/opt/spark-ds41f/candidate/vllm/tests/kernels/deepseek_v41_pdl_sources.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
ks = {
    "oneshot_allreduce": AR._build("bfloat16", 4, 0, 512, 2, 64, 2, False, False)(),
    "oneshot_allgather": AG._build(4, 0, 512, 2, 64, 2, False)(),
    "oneshot_reducescatter": RS._build("bfloat16", 4, 0, 512, 2, 64, 2, False)(),
    "oneshot_exchange": EX._build(4, 0, 512, 2, 64, 2, False)(),
}
for name, k in ks.items():
    src = k.get_kernel_source()
    open(f"/out/{name}.cu", "w").write(src)
    for fname, ptrs, body in ps.functions(src):
        if not fname.startswith(name):
            continue
        lines = [l.rstrip() for l in body.splitlines() if l.strip()]
        w = next(i for i, l in enumerate(lines) if ps.WAIT in l)
        first_ptr = next(i for i, l in enumerate(lines) if any(re.search(r"(?<![\w.])" + p + r"\b", l) for p in ptrs))
        print(f"== {fname}: pointer params {ptrs}")
        print(f"   {w} lines before the wait; wait at body line {w + 1}; first line naming a pointer param: body line {first_ptr + 1}")
        print("   before wait:", [l.strip() for l in lines[:w]])
        for l in lines[w:w + 4]:
            print("   |", l.strip()[:170])
