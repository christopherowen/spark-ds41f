"""pytest plugin: SWAP="indexer=/b/indexer_bits.py,activation=/b/activation_expr.py" serves
each file as vllm.models.deepseek_v4_1.tilelang.<module> before any test imports it."""
import importlib.util
import os
import sys

for item in filter(None, os.environ.get("SWAP", "").split(",")):
    name, path = item.split("=", 1)
    import vllm.models.deepseek_v4_1.tilelang as package

    full = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    setattr(package, name, module)
    print(f"swap_plugin: {full} is {path}", flush=True)
