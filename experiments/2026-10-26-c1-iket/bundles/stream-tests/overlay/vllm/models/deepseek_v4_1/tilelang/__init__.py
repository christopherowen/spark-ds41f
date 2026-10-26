# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 kernels in TileLang for SM120 / SM121:

- ``sparse_mla``: compressed sparse MLA over the SWA and indexed caches;
- ``indexer``: MXFP4 index scores, exact top-k and candidate lists;
- ``gemm``: MXFP8 block-scaled and BF16 GEMMs;
- ``moe``: routed-expert dispatch, grouped FP8 x FP4 GEMMs and combine;
- ``attention``, ``linear``, ``norm``: the vLLM layers and methods built on them.

Importing a kernel module requires TileLang; the layers also use DeepSeek's
TileKernels for MXFP8 casts, the fused SwiGLU and RMSNorm.
"""
