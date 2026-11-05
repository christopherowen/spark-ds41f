"""Run CPU graph/warmup checks from the actual source in a CUDA dependency image."""
import ast
from pathlib import Path
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
import triton
import triton.language
import vllm.triton_utils
vllm.triton_utils.triton = triton
vllm.triton_utils.tl = triton.language
import vllm.tilelang_utils
vllm.tilelang_utils._ensure_tilelang_imported()
# This file's module-level guard skips its CPU shape tests without a GPU.
# Extract only those unchanged test bodies/helpers; omit all device tests.
source=Path('tests/kernels/test_deepseek_v41_tilelang_dspark_context.py')
names={'_recipe_ladder','_decode_config','_no_graph_pool','_target_manager','test_context_capture_sizes_are_the_target_decode_graph_sizes'}
selected=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(selected)==len(names)
target=Path('/tmp/test_verification_context_shapes.py')
target.write_text('import pytest\nimport torch\n'+ast.unparse(ast.Module(body=selected,type_ignores=[]))+'\n')
import pytest
raise SystemExit(pytest.main(['--noconftest','tests/v1/cudagraph/test_cudagraph_manager.py','tests/model_executor/test_b12x_warmup.py',str(target),'-q','-k','not embedding_collectives_have_rank_stable_names']))
