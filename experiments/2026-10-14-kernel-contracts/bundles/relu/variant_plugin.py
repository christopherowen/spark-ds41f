"""pytest plugin: with VARIANT=<name>, /b/indexer_<name>.py is served as
vllm.models.deepseek_v4_1.tilelang.indexer before any test imports it, so the
existing indexer tests run unchanged against that variant."""
import importlib.util
import os
import sys

NAME = "vllm.models.deepseek_v4_1.tilelang.indexer"
VARIANT = os.environ.get("VARIANT")

if VARIANT:
    import vllm.models.deepseek_v4_1.tilelang as package

    spec = importlib.util.spec_from_file_location(NAME, f"/b/indexer_{VARIANT}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    package.indexer = module
    print(f"variant_plugin: indexer is /b/indexer_{VARIANT}.py", flush=True)
