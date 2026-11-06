"""CPU-only rehearsal of dual-module default comparison, without native compile."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tilelang
from tilelang import tvm
from vllm.models.deepseek_v4_1.tilelang import moe

path = Path('/experiment/bundles/swizzle-screen/control_moe.py')
spec = importlib.util.spec_from_file_location('vllm.models.deepseek_v4_1.tilelang._control_moe', path)
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)
target = tvm.target.Target({'kind': 'cuda', 'arch': 'sm_121'})
rows = []
for inter in (576, 768):
    old = control.MoeExpertsV41Plan(384, 6, 5120, inter, swiglu_limit=10.)
    new = moe.MoeExpertsV41Plan(384, 6, 5120, inter, swiglu_limit=10.)
    assert old.prefill_swizzle_panel == 3 and new.prefill_swizzle_panel == 0
    for name, call in old.kernels().items():
        if name.endswith('swiglu'):
            continue  # unchanged helper selects vector width from the real GPU
        sources = []
        for factory, args, kwargs in (call, new.kernels()[name]):
            tir = factory.get_tir(*args, **kwargs)
            with target, tvm.transform.PassContext(opt_level=3, config=factory.pass_configs):
                sources.append(tilelang.lower(tir, target=target, enable_device_compile=False).kernel_source)
        assert sources[0] == sources[1], (inter, name)
        rows.append({'intermediate': inter, 'kernel': name, 'sha256': hashlib.sha256(sources[0].encode()).hexdigest()})
print(json.dumps({'cpu_only': True, 'dual_module_default_matches': rows}, indent=2))
