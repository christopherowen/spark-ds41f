# Full TileLang: what is left

Owner, 2026-10-17: "we also want to go full tilelang and simplify things". An inventory
from the r6w profile (experiments/2026-10-18-r6w-swiglu, `profile.json`, rank 0 on dgx1;
26 one-stream steps and 110 sixteen-stream code steps), before any port is planned.

## Already TileLang

Every kernel named `<module>_<what>_kernel` (the routed experts' grouped GEMM and SwiGLU,
dispatch and combine, the dense and split-K GEMMs, sparse MLA, the indexer, metadata, the
cache writer, compression, RoPE, mHC, the router, the vocabulary heads, Engram, the L2
prefetch), TileKernels' casts and norms, and sparknet's one-shot all-reduce and all-gather
(`sparknet/oneshot/_oneshot_tilelang.py`, `_allgather_tilelang.py` in the r6v image). At
sixteen streams these are all but about 1.2 ms of a 71 ms step's kernel time.

## Not TileLang (sixteen code streams, ms a step of kernel time)

| What | ms/step | Where |
| --- | ---: | --- |
| NCCL SendRecv (the ring's row all-to-all, PyNCCL) | 0.56 | vLLM 0180 |
| NCCL AllGather and Broadcast | 0.22 | vLLM collectives |
| torch copies, index-select, fills | 0.27 | glue around the plans |
| Triton (`_partial_argmax*`, `_gather_block_tables_kernel`, `_apply_write_kernel`, `_zero_kv_blocks_kernel`, CED `_copy_rows`) | about 0.1 | sampling, block tables, KV writes, CED |
| `mhc_post_fwd_cuda_kernel` | 0.10 | TileKernels CUDA |
| vLLM `vocab_parallel_embedding_kernel` | small | embedding |

Beyond kernels, B12X remains a dependency for the checkpoint loader (`--load-format b12x`),
the drafter's `block_fp8_linear` (vllm/models/deepseek_v4_1/nvidia/dspark.py) and a 0.005 ms
GEMV.

## Reading

Compute is effectively TileLang already. The ceiling of porting what is left is the
non-TileLang kernel time, about 1.3 ms of 71 at sixteen streams (about 2%) and less at one
stream, and only if a port is faster; the simplification is mostly in dependencies (B12X's
loader and drafter linear, the NCCL path of the ring all-to-all) rather than in kernels.
Candidates, by size: the ring's row all-to-all as a sparknet TileLang collective (0.56
ms/step), the drafter's FP8 linear in TileLang (drops B12X from the forward pass), the torch
glue copies.
