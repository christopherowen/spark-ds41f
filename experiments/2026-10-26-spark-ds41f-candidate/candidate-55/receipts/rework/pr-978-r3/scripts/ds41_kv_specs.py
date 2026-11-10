#!/usr/bin/env python3
"""DS4.1's KV cache specs as the recipe's workers report them, without weights: the
VllmConfig from the served checkpoint with the recipe's cache/scheduler flags, the
model built on the meta device, then the worker's own get_kv_cache_spec. Writes a
pickle of the spec dict (and a JSON summary) for gate_replay.py."""
import json
import os
import pickle
import sys

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext

out = sys.argv[1]
SPEC = os.environ.get("GATE_SPEC", "dspark")


def engine_args():
    kw = dict(model=os.environ["DS41F_SNAPSHOT"], tokenizer_mode="deepseek_v41", dtype="bfloat16",
              block_size=256, enable_prefix_caching=True, enable_chunked_prefill=True,
              max_model_len=1048576, max_num_seqs=16, max_num_batched_tokens=16384,
              long_prefill_token_threshold=16384 - 16 * 6, async_scheduling=True,
              prefix_cache_retention_interval=8192, limit_mm_per_prompt={"image": 67},
              mm_processor_cache_gb=0, kv_cache_memory_bytes=11274289152,
              max_parallel_prefills=1, attention_backend="TILELANG",
              engram_config={"table_memory": "disk", "disk_resident_scales": False,
                             "projection_tp": True})
    if SPEC == "dspark":
        kw["speculative_config"] = {"method": "dspark", "num_speculative_tokens": 5,
                                    "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
                                    "draft_sample_method": "greedy", "rejection_sample_method": "block",
                                    "enable_adaptive_verification": True,
                                    "adaptive_verification_row_cost": 0.035}
    return EngineArgs(**kw)


def main():
    cfg = engine_args().create_engine_config(UsageContext.OPENAI_API_SERVER)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    with set_current_vllm_config(cfg):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method="tcp://127.0.0.1:29533", backend="gloo")
        initialize_model_parallel(1, 1)
        from vllm.model_executor.model_loader.utils import initialize_model
        with torch.device("meta"):
            model = initialize_model(vllm_config=cfg)
        drafter = None
        if cfg.speculative_config is not None:
            try:
                from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator  # noqa: F401
            except Exception as e:  # noqa: BLE001
                drafter = f"not built: {e!r}"
        from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
        specs = get_kv_cache_spec(cfg)
    summary = {}
    for name, s in specs.items():
        key = f"{type(s).__name__}(block={s.block_size}" + (
            f",window={getattr(s, 'sliding_window', None)}" if hasattr(s, "sliding_window") else "") + ")"
        summary[key] = summary.get(key, 0) + 1
    with open(out, "wb") as f:
        pickle.dump(specs, f)
    print(json.dumps({"layers": len(specs), "kinds": summary, "drafter": drafter,
                      "model": type(model).__name__}, indent=1))


if __name__ == "__main__":
    main()
