import torch

assert torch.cuda.is_available()
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import vllm.models.deepseek_v4_1.nvidia.dspark  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.model  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.vl_model  # noqa: F401
import vllm.v1.attention.backends.mla.b12x_mla_sparse  # noqa: F401
from vllm.models.deepseek_v41 import DeepseekV41ForCausalLM

print("DS4.1 SM12x class:", DeepseekV41ForCausalLM.__module__)
assert DeepseekV41ForCausalLM.__module__.startswith("vllm.models.deepseek_v4_1")
print("GPU import smoke passed")
import importlib.metadata

import tilelang
import tile_kernels  # noqa: F401

assert importlib.metadata.version("tilelang") == '0.1.15', tilelang.__version__
print("TileLang", tilelang.__version__)
import vllm.models.deepseek_v4_1.tilelang  # noqa: F401
print("TileLang import smoke passed")
from sparknet.integration.vllm import SparknetOneShotAllReduce  # noqa: F401
from sparknet.oneshot._proxy import load

assert importlib.metadata.version("dgx-spark-networking") == '0.5.0'
print("sparknet proxy ABI", load().roce_abi_version())
print("sparknet import smoke passed")
