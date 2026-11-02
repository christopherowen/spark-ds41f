"""DS4.1 decode steps end to end on one GPU, for IKET traces.

usage: iket_step.py off|on --out DIR [--iket DIR] [--ctas N] [--iters N] [--warps N]
                    [--granularity {fine,phase}] [--payloads] [--iket-cache-dir DIR]
                    [--skip REGEX] [--tp-shape 1|3|4] [--requests N] [--tokens N]
                    [--drafts N] [--grid N] [--grid-rows N] [--eager] [--filled]
                    [--batched N] [--chunk N] [--model DIR] [--overrides JSON]

``--tp-shape`` gives the layers one rank's shapes at TP4 (the ring) or TP3 (the
triangle); ``--requests`` is the measured batch (c1, c8); ``--grid N`` follows the
request with the engine grid's dummy step at its shape (servebench engine), between
``iket_grid_marker`` launches, for step_diff.py.

When ``--requests >= 8`` and ``--eager`` is not set, ``--grid-rows`` defaults to 2
verified rows per request (matching adaptive verification's 2 rows/request at c8 and the
recipe's 16-row CUDA graph capture size) unless overridden.
"""

import argparse
import functools
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path

MODEL = ("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
         "1bae0e1088993b9b82de08f59de5be5a34e42170")
TEXT_OVERRIDES = {
    "num_hidden_layers": 6,
    # Six target layers, then the drafter's three (ratio 0, as in the checkpoint).
    "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4],
    "candidate_source_layer_id": 4,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [3, 4, 5],
}
FULL_TEXT_OVERRIDES = {
    "num_hidden_layers": 40,
    # All 40 target layers (2 SWA, 18 CSA, 20 HCA) + 3 DSpark drafter SWA layers.
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [37, 38, 39],
}
TP_SHAPES = {
    1: {},
    3: {"num_attention_heads": 24, "o_groups": 3, "moe_intermediate_size": 768},
    4: {"num_attention_heads": 16, "o_groups": 2, "moe_intermediate_size": 576},
}
PROMPTS = [
    "Explain in detail how a sparse mixture-of-experts layer routes each token to its experts, "
    "and why the routing has to be balanced across the experts during training.",
    "Write a short story about a lighthouse keeper who discovers that the light has been "
    "signalling to ships that do not exist, and what she decides to do about it.",
]


def capture_ladder(max_size):
    """The recipe's CUDA-graph capture sizes up to ``max_size`` (scripts/recipe.py
    ``capture_sizes``): 1-4, 6, 8, every 4th to 32, then every 8th, ending at
    ``max_size``."""
    ladder = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, max_size + 1, 8)]
    return [*(size for size in ladder if size < max_size), max_size]


def default_grid_rows(requests: int, drafts: int, grid_rows: int = 0, eager: bool = False) -> int:
    """Verified rows per request for the engine grid's dummy step.
    At c8 with CUDA graphs and adaptive verification enabled, adaptive verification
    verifies 2 rows per request (16 rows total), replaying the 16-row graph."""
    if grid_rows > 0:
        return grid_rows
    if requests >= 8 and not eager:
        return 2
    return drafts + 1


def shrink(config, overrides):
    """The six-layer config with ``overrides``, for the target and the DSpark drafter."""
    text = config.get_text_config()
    for key, value in overrides.items():
        setattr(text, key, value)
    return config


def same_dummy_weights(uint8_fill=0):
    """vLLM's dummy loader seeds floating tensors but leaves integer ones (the packed FP4
    experts and their UE8M0 scales) as allocated, different in every process. Zero them,
    as it does on ROCm, so that every process loads the same weights and runs' tokens
    can be compared bit for bit. Uses cudaMemsetAsync instead of ATen FillFunctor so
    run-iket --warp-trace does not record 151M weight-init warps (3.64 GB of trace buffer)
    across 43 layers."""
    import ctypes
    import torch
    from vllm.model_executor.model_loader import dummy_loader, weight_utils

    cudart = None
    for libname in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(libname)
            cudart.cudaMemsetAsync.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
            ]
            cudart.cudaMemsetAsync.restype = ctypes.c_int
            cudart.cudaMemcpyAsync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            cudart.cudaMemcpyAsync.restype = ctypes.c_int
            break
        except OSError:
            cudart = None

    seeded = weight_utils.initialize_single_dummy_weight
    init_cache = {}

    def initialize(param, *args, **kwargs):
        if param.device.type == "meta":
            return
        dummy_val = getattr(param, "dummy_weight_value", None)
        if (not torch.is_floating_point(param) and dummy_val is None) or dummy_val == 0:
            fill_byte = (
                int(uint8_fill) & 0xFF
                if (param.dtype == torch.uint8 and dummy_val is None)
                else 0
            )
            if cudart is not None and param.is_cuda and param.is_contiguous():
                stream = torch.cuda.current_stream(param.device).cuda_stream
                rc = cudart.cudaMemsetAsync(
                    ctypes.c_void_p(param.data_ptr()),
                    fill_byte,
                    param.numel() * param.element_size(),
                    ctypes.c_void_p(stream),
                )
                if rc == 0:
                    return
            if param.dtype == torch.uint8 and dummy_val is None:
                param.fill_(uint8_fill)
            else:
                param.zero_()
            return
        nbytes = param.numel() * param.element_size()
        if (
            cudart is not None
            and param.is_cuda
            and param.is_contiguous()
            and nbytes >= 4096
        ):
            key = (
                tuple(param.shape),
                param.dtype,
                dummy_val,
                args,
                tuple(sorted(kwargs.items())),
                str(param.device),
            )
            cached = init_cache.get(key)
            if cached is not None and cached.data_ptr() != param.data_ptr():
                stream = torch.cuda.current_stream(param.device).cuda_stream
                rc = cudart.cudaMemcpyAsync(
                    ctypes.c_void_p(param.data_ptr()),
                    ctypes.c_void_p(cached.data_ptr()),
                    nbytes,
                    3,  # cudaMemcpyDeviceToDevice
                    ctypes.c_void_p(stream),
                )
                if rc == 0:
                    return
            seeded(param, *args, **kwargs)
            init_cache[key] = param
            return
        seeded(param, *args, **kwargs)

    dummy_loader.initialize_single_dummy_weight = initialize
    weight_utils.initialize_single_dummy_weight = initialize

    orig_zeros = torch.zeros

    def fast_zeros(*size, **kwargs):
        if (
            cudart is None
            or kwargs.get("out") is not None
            or torch.compiler.is_compiling()
        ):
            return orig_zeros(*size, **kwargs)
        out = torch.empty(*size, **kwargs)
        if out.is_cuda and out.is_contiguous():
            nbytes = out.numel() * out.element_size()
            if nbytes >= 65536:
                stream = torch.cuda.current_stream(out.device).cuda_stream
                rc = cudart.cudaMemsetAsync(
                    ctypes.c_void_p(out.data_ptr()),
                    0,
                    nbytes,
                    ctypes.c_void_p(stream),
                )
                if rc == 0:
                    return out
        return orig_zeros(*size, **kwargs)

    torch.zeros = fast_zeros

    orig_zero_ = torch.Tensor.zero_

    def fast_zero_(self):
        if (
            cudart is not None
            and not torch.compiler.is_compiling()
            and self.is_cuda
            and self.is_contiguous()
        ):
            nbytes = self.numel() * self.element_size()
            if nbytes >= 65536:
                stream = torch.cuda.current_stream(self.device).cuda_stream
                rc = cudart.cudaMemsetAsync(
                    ctypes.c_void_p(self.data_ptr()),
                    0,
                    nbytes,
                    ctypes.c_void_p(stream),
                )
                if rc == 0:
                    return self
        return orig_zero_(self)

    torch.Tensor.zero_ = fast_zero_

    def clone_dma(src):
        if src is None:
            return None
        dst = torch.empty_like(src)
        nbytes = dst.numel() * dst.element_size()
        if nbytes > 0 and cudart is not None and dst.is_cuda and dst.is_contiguous():
            stream = torch.cuda.current_stream(dst.device).cuda_stream
            rc = cudart.cudaMemcpyAsync(
                ctypes.c_void_p(dst.data_ptr()),
                ctypes.c_void_p(src.data_ptr()),
                nbytes,
                3,  # cudaMemcpyDeviceToDevice
                ctypes.c_void_p(stream),
            )
            if rc == 0:
                return dst
        dst.copy_(src)
        return dst

    from vllm.models.deepseek_v4_1.tilelang import (
        gemm as tl_gemm,
        linear as tl_linear,
        moe as tl_moe,
    )

    orig_prepare_weights = tl_moe.MoeExpertsV41Plan.prepare_weights
    moe_prep_cache = {}

    def fast_prepare_weights(self, w13, w13_scale, w2, w2_scale):
        key = (
            tuple(w13.shape),
            w13.dtype,
            tuple(w13_scale.shape),
            w13_scale.dtype,
            tuple(w2.shape),
            w2.dtype,
            tuple(w2_scale.shape),
            w2_scale.dtype,
            self.block_N,
            str(w13.device),
        )
        cached = moe_prep_cache.get(key)
        if cached is None:
            res = orig_prepare_weights(self, w13, w13_scale, w2, w2_scale)
            moe_prep_cache[key] = res
            return res
        cloned = []
        for tiles, tails, codes, _codes_2bit, narrow, base in cached:
            new_codes = clone_dma(codes)
            cloned.append(
                (
                    clone_dma(tiles),
                    clone_dma(tails),
                    new_codes,
                    new_codes.view(torch.uint8),
                    clone_dma(narrow),
                    clone_dma(base),
                )
            )
        return tuple(cloned)

    tl_moe.MoeExpertsV41Plan.prepare_weights = fast_prepare_weights

    orig_pack_scale_words = tl_linear.pack_scale_words
    scale_words_cache = {}

    def fast_pack_scale_words(scales, *args, **kwargs):
        key = (
            tuple(scales.shape),
            scales.dtype,
            str(scales.device),
            args,
            tuple(sorted(kwargs.items())),
        )
        cached = scale_words_cache.get(key)
        if cached is None:
            res = orig_pack_scale_words(scales, *args, **kwargs)
            scale_words_cache[key] = res
            return res
        return clone_dma(cached)

    tl_linear.pack_scale_words = fast_pack_scale_words
    tl_gemm.pack_scale_words = fast_pack_scale_words

    def cleanup():
        init_cache.clear()
        moe_prep_cache.clear()
        scale_words_cache.clear()
        torch.zeros = orig_zeros
        torch.Tensor.zero_ = orig_zero_
        dummy_loader.initialize_single_dummy_weight = seeded
        weight_utils.initialize_single_dummy_weight = seeded
        tl_moe.MoeExpertsV41Plan.prepare_weights = orig_prepare_weights
        tl_linear.pack_scale_words = orig_pack_scale_words
        tl_gemm.pack_scale_words = orig_pack_scale_words

    return cleanup


def _enable_lean_capture_filter(requests: int) -> None:
    """When ``--lean-capture`` is enabled, restrict variable-length FULL CUDA-graph
    capture shapes, sampler warmup request counts, and eager tail-shape warmups to
    ``num_reqs <= requests`` so startup does not record dozens of unused 43-layer
    passes into the ``run-iket`` trace buffer."""
    import vllm.v1.worker.gpu.cudagraph_utils as cg_utils
    import vllm.v1.worker.gpu.spec_decode.adaptive_verification as av_mod
    import vllm.v1.worker.gpu.warmup as warmup_mod

    orig_dense_shapes = cg_utils.dense_varlen_decode_shapes

    def lean_dense_varlen_decode_shapes(max_num_reqs: int, decode_query_len: int, max_capture_tokens: int):
        allowed_q = {1, 3, decode_query_len}
        return tuple(
            (num_reqs, num_reqs * q)
            for num_reqs, q_tokens in orig_dense_shapes(
                min(max_num_reqs, requests), decode_query_len, max_capture_tokens
            )
            for q in (q_tokens // max(1, num_reqs),)
            if num_reqs == requests and q in allowed_q
        )

    cg_utils.dense_varlen_decode_shapes = lean_dense_varlen_decode_shapes

    orig_cg_init = cg_utils.CudaGraphManager.__init__

    def lean_cg_init(self, vllm_config, *args, **kwargs):
        sched = getattr(vllm_config, "scheduler_config", None)
        orig_max_seqs = getattr(sched, "max_num_seqs", None)
        # Keep DSparkContextCudaGraphs (decode_query_len == 1, varlen_decode == False)
        # able to capture its power-of-two row capacities up to 6, while capping
        # request-indexed CudaGraphManagers to `requests`.
        decode_query_len = args[2] if len(args) >= 3 else kwargs.get("decode_query_len", 1)
        varlen = kwargs.get("varlen_decode", args[4] if len(args) >= 5 else False)
        cap_reqs = varlen or decode_query_len > 1
        try:
            if cap_reqs and sched is not None and orig_max_seqs is not None:
                sched.max_num_seqs = min(orig_max_seqs, requests)
            orig_cg_init(self, vllm_config, *args, **kwargs)
        finally:
            if cap_reqs and sched is not None and orig_max_seqs is not None:
                sched.max_num_seqs = orig_max_seqs

    cg_utils.CudaGraphManager.__init__ = lean_cg_init

    orig_warmup_shape = warmup_mod.warmup_prefill_shape

    def lean_warmup_prefill_shape(*, max_num_seqs: int, max_num_batched_tokens: int, decode_query_len: int):
        return orig_warmup_shape(
            max_num_seqs=min(max_num_seqs, requests),
            max_num_batched_tokens=max_num_batched_tokens,
            decode_query_len=decode_query_len,
        )

    warmup_mod.warmup_prefill_shape = lean_warmup_prefill_shape
    av_mod.preparation_tail_sizes = lambda capture_sizes, max_num_tokens: ()
    if requests < 2:
        warmup_mod.run_mixed_prefill_decode_warmup = lambda *args, **kwargs: False

    try:
        import vllm.model_executor.warmup.kernel_warmup as kw_mod

        kw_mod.flashinfer_autotune = lambda *args, **kwargs: None
    except Exception:
        pass

    try:
        import torch
        import vllm.v1.worker.gpu.model_runner as mr_mod
        from vllm.v1.worker.workspace import current_workspace_manager, use_workspace_lane

        orig_profile_run = mr_mod.GPUModelRunner.profile_run
        orig_profile_dsv4_attn = mr_mod.GPUModelRunner._profile_deepseek_v4_attention

        def lean_profile_dsv4_attn(self, prepare_profile_state=None):
            if getattr(self.cache_config, "kv_cache_memory_bytes", None):
                return
            return orig_profile_dsv4_attn(self, prepare_profile_state)

        def lean_profile_run(self, prepare_profile_state=None):
            if not getattr(self.cache_config, "kv_cache_memory_bytes", None):
                return orig_profile_run(self, prepare_profile_state)
            orig_tokens = self.max_num_tokens
            # Pre-reserve full max_num_tokens mHC and extend-attention workspace lanes
            # so capping the dummy forward pass token count does not shrink workspaces.
            hidden = getattr(self.model_config.hf_text_config, "hidden_size", 4096)
            splits = max(1, hidden // 512)
            mhc_specs = (
                ((splits, orig_tokens, 1, 3), torch.float32),
                ((splits, orig_tokens, 1), torch.float32),
                ((orig_tokens, hidden), torch.bfloat16),
                ((splits, orig_tokens), torch.float32),
            )
            for lane in (0, getattr(self, "_draft_workspace_lane", 1)):
                with use_workspace_lane(lane):
                    current_workspace_manager().get_simultaneous(*mhc_specs)
                    for module in self.compilation_config.static_forward_context.values():
                        attn_ws = getattr(module, "_attention_workspace_specs", None)
                        attn_sc = getattr(module, "_attention_scratch_specs", None)
                        if isinstance(attn_ws, dict) and "extend" in attn_ws and callable(attn_sc):
                            try:
                                current_workspace_manager().get_simultaneous(
                                    *attn_ws["extend"], *attn_sc("extend", orig_tokens)
                                )
                            except Exception:
                                pass
            try:
                self.max_num_tokens = min(orig_tokens, max(64, requests * 8))
                return orig_profile_run(self, prepare_profile_state)
            finally:
                self.max_num_tokens = orig_tokens

        mr_mod.GPUModelRunner._profile_deepseek_v4_attention = lean_profile_dsv4_attn
        mr_mod.GPUModelRunner.profile_run = lean_profile_run
    except Exception:
        pass


def nan_hooks(worker):
    """Run on the worker (collective_rpc): a forward hook on every module of the model
    that prints the first module whose output is not finite, and any embedding's
    out-of-range token ids. Hooks run in eager mode only (not in a CUDA graph)."""
    import torch

    model = worker.model_runner.model
    first = []

    def hook(module, inputs, output):
        if first:
            return
        weight = getattr(module, "weight", None)
        if isinstance(module, torch.nn.Embedding) or (
                inputs and isinstance(inputs[0], torch.Tensor) and not inputs[0].is_floating_point()
                and weight is not None and weight.dim() == 2):
            ids = inputs[0]
            if ids.numel() and (int(ids.max()) >= weight.shape[0] or int(ids.min()) < 0):
                first.append(module.nan_name)
                print(f"out-of-range ids into {module.nan_name}: min {int(ids.min())} max "
                      f"{int(ids.max())} rows {weight.shape[0]}", flush=True)
                return
        for out in output if isinstance(output, (tuple, list)) else (output,):
            if isinstance(out, torch.Tensor) and out.is_floating_point() and out.numel() \
                    and not bool(torch.isfinite(out).all()):
                first.append(module.nan_name)
                bad = (~torch.isfinite(out)).nonzero()
                rows = sorted({int(r) for r in bad[:, 0].tolist()})
                print(f"first non-finite output: {module.nan_name} ({type(module).__name__}) "
                      f"shape {tuple(out.shape)}, {bad.shape[0]} values, rows {rows[0]}-{rows[-1]} "
                      f"({len(rows)})", flush=True)
                for i, x in enumerate(inputs):
                    if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dim() >= 2:
                        sel = x[rows].float()
                        print(f"  its input {i} {tuple(x.shape)} {x.dtype}, those rows: finite "
                              f"{bool(torch.isfinite(sel).all())}, absmax {float(sel.abs().max()):.3g}, "
                              f"all-zero rows {int((sel.abs().amax(dim=-1) == 0).sum())}; other rows' "
                              f"absmax {float(x.float().abs().max()):.3g}", flush=True)
                return

    def finite(value):
        return not (isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel()
                    and not bool(torch.isfinite(value).all()))

    from vllm.v1.worker.workspace import current_workspace_manager

    aliased = []

    def in_workspace(tensor):
        start = tensor.data_ptr()
        for lane, buffer in enumerate(current_workspace_manager()._current_workspaces):
            if buffer is not None:
                low = buffer.data_ptr()
                if low <= start < low + buffer.numel() * buffer.element_size():
                    return lane
        return None

    def checked(label, method):
        def call(*args, **kwargs):
            previous = kwargs.get("previous_output", args[0] if label.endswith("post_pre") and args else None)
            if isinstance(previous, torch.Tensor) and not aliased:
                lane = in_workspace(previous)
                if lane is not None:
                    aliased.append(label)
                    print(f"{label}: previous_output {tuple(previous.shape)} lies in workspace lane "
                          f"{lane} (rows {previous.shape[0]})", flush=True)
            out = method(*args, **kwargs)
            if not first:
                bad = [i for i, o in enumerate(out if isinstance(out, tuple) else (out,)) if not finite(o)]
                if bad:
                    first.append(label)
                    inputs = {**{str(i): a for i, a in enumerate(args)}, **kwargs}
                    print(f"first non-finite output: {label} outputs {bad}; non-finite inputs "
                          f"{[k for k, v in inputs.items() if not finite(v)]}; tensor inputs "
                          f"{[(k, tuple(v.shape)) for k, v in inputs.items() if isinstance(v, torch.Tensor)]}",
                          flush=True)
            return out
        return call

    for name, module in model.named_modules():
        module.nan_name = name or "model"
        module.register_forward_hook(hook)
        mhc = getattr(module, "mhc", None)
        for method in ("pre", "post_pre"):
            if mhc is not None and callable(getattr(mhc, method, None)):
                setattr(mhc, method, checked(f"{name}.mhc.{method}", getattr(mhc, method)))
    return len(list(model.named_modules()))


def marker_kernels():
    """Two one-thread kernels whose launches bracket the measured request
    (``iket_step_marker``) and the engine grid's steps (``iket_grid_marker``)."""
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def iket_step_marker(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            A[0] = A[0] + 1

    @T.prim_func
    def iket_grid_marker(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            A[0] = A[0] + 2

    return (tilelang.compile(iket_step_marker, target="cuda"),
            tilelang.compile(iket_grid_marker, target="cuda"))


def _enable_iket(args):
    os.environ["VLLM_DS41_IKET"] = args.iket
    os.environ["VLLM_DS41_IKET_CTAS"] = str(args.ctas)
    os.environ["VLLM_DS41_IKET_ITERS"] = str(args.iters)
    os.environ["VLLM_DS41_IKET_WARPS"] = str(args.warps)
    os.environ["VLLM_DS41_IKET_GRANULARITY"] = args.granularity
    os.environ["VLLM_DS41_IKET_PAYLOADS"] = "1" if args.payloads else "0"
    os.environ["VLLM_DS41_IKET_SKIP"] = args.skip
    if args.iket_cache_dir:
        os.environ["VLLM_DS41_IKET_CACHE_DIR"] = args.iket_cache_dir

    from vllm.models.deepseek_v4_1.tilelang import iket

    sig = inspect.signature(iket.enable).parameters
    kwargs = {}
    if "warps" in sig:
        kwargs["warps"] = args.warps
    if "granularity" in sig:
        kwargs["granularity"] = args.granularity
    if "payloads" in sig:
        kwargs["payloads"] = args.payloads
    if "cache_dir" in sig and args.iket_cache_dir:
        kwargs["cache_dir"] = args.iket_cache_dir
    iket.enable(args.iket, args.ctas, args.iters, args.skip, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("off", "on"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--iket")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--drafts", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1,
                        help="concurrent requests in the measured batch (c1, c8)")
    parser.add_argument("--grid", type=int, default=0,
                        help="after the request, the engine grid's dummy step (servebench engine) "
                             "at its requests, verification rows and context: one warm-up and this "
                             "many timed runs, between iket_grid_marker launches")
    parser.add_argument("--grid-rows", type=int, default=0,
                        help="the grid's verified rows a request (0: 2 at c8+ with graphs, else 1 + drafts)")
    parser.add_argument("--ctas", type=int, default=0, help="VLLM_DS41_IKET_CTAS")
    parser.add_argument("--iters", type=int, default=0, help="VLLM_DS41_IKET_ITERS")
    parser.add_argument("--warps", type=int, default=0, help="VLLM_DS41_IKET_WARPS")
    parser.add_argument("--granularity", choices=("fine", "phase"), default="phase",
                        help="VLLM_DS41_IKET_GRANULARITY")
    parser.add_argument("--payloads", action="store_true", help="VLLM_DS41_IKET_PAYLOADS")
    parser.add_argument("--iket-cache-dir", default="", help="VLLM_DS41_IKET_CACHE_DIR")
    parser.add_argument("--skip", default="", help="VLLM_DS41_IKET_SKIP")
    parser.add_argument("--eager", action="store_true",
                        help="no CUDA graphs (enforce_eager): every kernel is a launch of its own")
    parser.add_argument("--batched", type=int, default=512, help="max_num_batched_tokens")
    parser.add_argument("--chunk", type=int, default=256, help="long_prefill_token_threshold")
    parser.add_argument("--max-num-seqs", type=int, default=0,
                        help="max_num_seqs (0: the requests, at least two); the recipe's is 16 "
                             "(TP4) or 8 (TP3), which sizes the per-request state the step reads")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="max_model_len; the recipe's 1,048,576 widens each block-table row "
                             "the step copies")
    parser.add_argument("--kv-bytes", type=int, default=2 << 30, help="kv_cache_memory_bytes")
    parser.add_argument("--prefix-caching", action="store_true",
                        help="enable_prefix_caching, as the recipe serves")
    parser.add_argument("--capture-ladder", action="store_true",
                        help="the recipe's CUDA-graph capture sizes (scripts/recipe.py "
                             "capture_sizes: 1-4, 6, 8, every 4th to 32, every 8th to the "
                             "streams' rows) instead of the measured batches' alone")
    parser.add_argument("--lean-capture", action="store_true",
                        help="capture only the exact decode CUDA-graph batch size replayed by "
                             "the measured batch (e.g. [6] at c1) to minimize non-measured warps")
    parser.add_argument("--no-warmup-request", action="store_true",
                        help="skip the unbracketed 16-token warmup request before iket_step_marker "
                             "(as specified in full-pass-spec.md)")
    parser.add_argument("--full-layers", action="store_true",
                        help="run all 40 target layers + 3 DSpark drafter layers instead of the "
                             "6-layer subset")
    parser.add_argument("--filled", action="store_true",
                        help="fill uninitialized tensors (torch's deterministic mode), for "
                             "token comparisons between processes; adds fill kernels")
    parser.add_argument("--nan-hooks", action="store_true",
                        help="with --eager: print the first module whose output is not finite")
    parser.add_argument("--uint8-fill", type=lambda text: int(text, 0), default=0,
                        help="the byte uint8 dummy weights hold (0: zero; 0x77: small nonzero experts)")
    parser.add_argument("--tp-shape", type=int, choices=sorted(TP_SHAPES), default=1,
                        help="one rank's shapes at this TP size (TP_SHAPES)")
    parser.add_argument("--overrides", default="{}",
                        help="JSON: text-config overrides applied after the layer set "
                             "and --tp-shape")
    args = parser.parse_args()
    base_overrides = FULL_TEXT_OVERRIDES if args.full_layers else TEXT_OVERRIDES
    overrides = {**base_overrides, **TP_SHAPES[args.tp_shape], **json.loads(args.overrides)}
    if overrides.get("num_hidden_layers", 6) >= 40 and os.environ.get("VLLM_LATE_ADMISSION_LAYER") in (None, "4"):
        os.environ["VLLM_LATE_ADMISSION_LAYER"] = "30"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    torch.cuda.init()
    if args.filled:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
    if args.mode == "on":
        _enable_iket(args)
    marker, grid_marker = marker_kernels()
    count = torch.zeros(1, dtype=torch.int32, device="cuda")

    from vllm import LLM, SamplingParams

    cleanup_dummy = same_dummy_weights(args.uint8_fill)
    if args.lean_capture:
        _enable_lean_capture_filter(args.requests)
    started = time.monotonic()
    llm = LLM(
        model=args.model,
        tokenizer_mode="deepseek_v41",
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=1,
        block_size=256,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs or max(2, args.requests),
        max_num_batched_tokens=args.batched,
        long_prefill_token_threshold=args.chunk,
        max_parallel_prefills=1,
        async_scheduling=True,
        kv_cache_memory_bytes=args.kv_bytes,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=args.prefix_caching,
        enforce_eager=args.eager,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        speculative_config={
            "method": "dspark", "num_speculative_tokens": args.drafts,
            "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
            "draft_sample_method": "greedy", "rejection_sample_method": "block",
            **({} if args.eager else {"enable_adaptive_verification": True,
                                      "adaptive_verification_row_cost": 0.035}),
        },
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
            "cudagraph_capture_sizes": (
                capture_ladder((args.max_num_seqs or max(2, args.requests)) * (args.drafts + 1))
                if args.capture_ladder
                else (
                    sorted({
                        args.requests * (args.drafts + 1),
                        args.requests * default_grid_rows(args.requests, args.drafts, args.grid_rows, args.eager),
                    })
                    if args.lean_capture
                    else sorted({n * (args.drafts + 1) for n in (1, 2, args.requests)})
                )),
        },
        generation_config="vllm",
        hf_overrides=functools.partial(shrink, overrides=overrides),
        jit_monitor_mode="off",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
    )
    cleanup_dummy()
    boot = time.monotonic() - started
    print(f"engine ready in {boot:.0f} s", flush=True)
    if args.nan_hooks:
        print(f"nan hooks on {llm.collective_rpc(nan_hooks)[0]} modules", flush=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True, logprobs=5)
    timings = []
    if not args.no_warmup_request:
        started = time.monotonic()
        llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
        timings.append(time.monotonic() - started)
    marker(count)
    torch.cuda.synchronize()
    started = time.monotonic()
    prompts = [PROMPTS[1]] + [f"{PROMPTS[1]} Version {i + 1}." for i in range(1, args.requests)]
    outputs = llm.generate(prompts, greedy)
    measured = outputs[0]
    torch.cuda.synchronize()
    timings.append(time.monotonic() - started)
    marker(count)
    torch.cuda.synchronize()
    grid = None
    if args.grid:
        context = len(measured.prompt_token_ids) + args.tokens // 2
        eff_rows = default_grid_rows(args.requests, args.drafts, args.grid_rows, args.eager)
        spec = {"shapes": [{"num_reqs": args.requests, "context_len": context,
                            "num_tokens": args.requests * eff_rows}],
                "replays": args.grid, "warmup": 1, "settle": 0}
        grid_marker(count)
        torch.cuda.synchronize()
        reply = llm.collective_rpc("engine_grid", args=(json.dumps(spec),))[0]["shapes"][0]
        torch.cuda.synchronize()
        grid_marker(count)
        torch.cuda.synchronize()
        grid = {"spec": spec, "runs": 1 + args.grid, "grid_rows_per_req": eff_rows,
                **{key: reply[key] for key in ("step_ms", "forward_ms", "drafter_ms")}}

    from vllm import envs

    result = {
        "mode": args.mode,
        "boot_s": round(boot, 1),
        "request_s": [round(t, 3) for t in timings],
        "requests": args.requests,
        "tokens": args.tokens,
        "prompt_tokens": len(measured.prompt_token_ids),
        "token_ids": list(measured.outputs[0].token_ids),
        "generated": [len(output.outputs[0].token_ids) for output in outputs],
        "logprobs": [{str(t): float(p.logprob) for t, p in sorted(step.items())}
                     for step in measured.outputs[0].logprobs or []],
        "batched": args.batched,
        "chunk": args.chunk,
        "max_num_seqs": args.max_num_seqs or max(2, args.requests),
        "max_model_len": args.max_model_len,
        "kv_bytes": args.kv_bytes,
        "prefix_caching": args.prefix_caching,
        "capture_ladder": args.capture_ladder,
        "lean_capture": args.lean_capture,
        "no_warmup_request": args.no_warmup_request,
        "full_layers": args.full_layers,
        "environment": {name: os.environ.get(name) for name in (
            "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_LATE_ADMISSION_LAYER", "CUDA_DEVICE_MAX_CONNECTIONS",
            "VLLM_DS41_KERNEL_BACKEND")},
        **{name: getattr(envs, name, os.environ.get(name)) for name in (
            "VLLM_DS41_IKET", "VLLM_DS41_IKET_CTAS", "VLLM_DS41_IKET_ITERS",
            "VLLM_DS41_IKET_WARPS", "VLLM_DS41_IKET_GRANULARITY",
            "VLLM_DS41_IKET_PAYLOADS", "VLLM_DS41_IKET_CACHE_DIR", "VLLM_DS41_IKET_SKIP")},
        "filled": args.filled,
        "eager": args.eager,
        "iket_modules": sorted(m for m in sys.modules if m.endswith(".iket")),
        "grid": grid,
        "tp_shape": args.tp_shape,
        "uint8_fill": args.uint8_fill,
        "text_overrides": overrides,
        "drafts": args.drafts,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    free = torch.cuda.mem_get_info()[0]
    llm.llm_engine.engine_core.shutdown()
    del llm, measured, outputs
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
    for obj in list(gc.get_objects()):
        try:
            if isinstance(obj, torch.cuda.CUDAGraph):
                obj.reset()
        except Exception:
            pass
    gc.collect()
    for obj in list(gc.get_objects()):
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                storage = obj.untyped_storage()
                if storage.size() > 0:
                    storage.resize_(0)
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    result["freed_gib_before_exit"] = round((torch.cuda.mem_get_info()[0] - free) / 2**30, 1)
    result["allocated_gib_at_exit"] = round(torch.cuda.memory_allocated() / 2**30, 1)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"engine shut down: {result['freed_gib_before_exit']} GiB freed, "
          f"{result['allocated_gib_at_exit']} GiB still allocated", flush=True)


if __name__ == "__main__":
    main()
