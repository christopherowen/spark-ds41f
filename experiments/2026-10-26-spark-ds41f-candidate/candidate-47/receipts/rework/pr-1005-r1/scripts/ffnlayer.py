"""Build DeepSeek V4.1 decoder layers' FFN half from the serving code, for timing.

The layers are vLLM's own ``DeepseekV4DecoderLayer`` (vllm/models/deepseek_v4_1/
nvidia/model.py), built under a VllmConfig made by ``EngineArgs`` from the served
checkpoint's config.json and the serving flags that reach the FFN (TileLang linear,
MoE and attention backends, V2 model runner, max_num_batched_tokens 16384). Only the
attention module is replaced (by a stub that owns nothing), because the FFN half does
not touch it. Single rank: TP4's / TP3's per-rank shapes are made by overriding
``moe_intermediate_size`` to the per-rank width (2304 / 4 = 576, 2304 / 3 = 768), so
the routed experts (I = 576 / 768), the shared expert (one expert: I = 576 / 768) and
the router (5120 -> 384, top-6) have exactly a serving rank's shapes; the cross-rank
all-reduce after the FFN is not part of the sublayer and is absent.

Weights are random with the checkpoint's dtypes (E4M3 + UE8M0 block-32 FP8 for the
shared expert, packed E2M1 + UE8M0 for the routed experts, BF16 / FP32 for the gate,
norms and mHC) and go through the serving ``process_weights_after_loading`` and
``prepare.declare_modules``, as the worker does after loading.

The FFN sublayer is the second half of ``DeepseekV4DecoderLayer.forward``:
``mhc.post_pre`` and ``ffn(x, input_ids)`` inside ``kernels.mxfp8_inputs(x_q)``,
under a forward context with the V2 runner's ``is_padding`` mask.
"""

import os
import types

import torch
from torch import nn

MODEL = "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
DEV = torch.device("cuda")


class _Segments:
    def l2_segments(self):
        return []


class StubAttention(nn.Module):
    """Stands in for the layer's attention: the FFN half never calls it."""

    def __init__(self, vllm_config, prefix="", topk_indices_buffer=None, aux_stream_list=None):
        super().__init__()
        self.fused_wqa_wkv = types.SimpleNamespace()
        self.is_ced_decoder = False
        self.compressor = None
        self._wo_projection_weights = _Segments()
        self._l2pf_wo = None
        self._l2pf_comp = None
        self._l2pf_ffn = None


_DIST = False


def make_config(intermediate: int):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model=MODEL,
        tokenizer=MODEL,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        load_format="dummy",
        max_model_len=65536,
        max_num_seqs=16,
        max_num_batched_tokens=16384,
        block_size=256,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]},
        hf_overrides={"moe_intermediate_size": intermediate},
        enforce_eager=False,
    )
    return args.create_engine_config()


def init_dist(vc):
    global _DIST
    if _DIST:
        return
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    with set_current_vllm_config(vc):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29517",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(DEV)
    _DIST = True


def _fill(name, p, g):
    t = p.data
    if t.dtype == torch.float8_e4m3fn:
        t.copy_((torch.randn(t.shape, generator=g, device=DEV) * 0.6).to(t.dtype))
    elif t.dtype == torch.float8_e8m0fnu:
        # UE8M0 block scales around 2^-7 .. 2^-5 (biased exponents 120..122).
        t.view(torch.uint8).copy_(
            torch.randint(120, 123, t.shape, generator=g, device=DEV, dtype=torch.int32).to(torch.uint8)
        )
    elif t.dtype in (torch.uint8, torch.int8):
        if "scale" in name:
            t.view(torch.uint8).copy_(
                torch.randint(120, 123, t.shape, generator=g, device=DEV, dtype=torch.int32).to(torch.uint8)
            )
        else:
            # packed E2M1 pairs: every nibble is a finite value
            t.view(torch.uint8).copy_(
                torch.randint(0, 256, t.shape, generator=g, device=DEV, dtype=torch.int32).to(torch.uint8)
            )
    elif t.dtype in (torch.int32, torch.int64):
        if "tid2eid" in name:
            raise RuntimeError("hash layer")
        t.zero_()
    elif t.is_floating_point():
        if name.endswith("norm.weight"):
            t.copy_(1.0 + 0.05 * torch.randn(t.shape, generator=g, device=DEV))
        elif "bias" in name:
            t.copy_(0.01 * torch.randn(t.shape, generator=g, device=DEV))
        elif name.startswith("hc_"):
            t.copy_(0.02 * torch.randn(t.shape, generator=g, device=DEV))
        else:
            fan_in = t.shape[-1] if t.ndim > 1 else 1
            t.copy_(torch.randn(t.shape, generator=g, device=DEV) / max(fan_in, 1) ** 0.5)
    else:
        raise RuntimeError(f"unhandled {name} {t.dtype}")


def build_layers(vc, n: int, first_index: int = 10, seed: int = 1, verbose=False):
    """n consecutive serving decoder layers (FFN half live), weights processed."""
    import vllm.models.deepseek_v4_1.nvidia.model as m
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.utils.torch_utils import set_default_torch_dtype

    m._select_dsv4_attn_cls = lambda _vc: StubAttention
    layers = []
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), torch.device(DEV):
        for i in range(n):
            layer = m.DeepseekV4DecoderLayer(vc, prefix=f"model.layers.{first_index + i}")
            g = torch.Generator(device=DEV)
            g.manual_seed(seed * 1000 + i)
            for name, p in layer.named_parameters():
                if verbose and i == 0:
                    print("param", name, tuple(p.shape), p.dtype)
                _fill(name, p, g)
            for name, b in layer.named_buffers():
                if verbose and i == 0:
                    print("buffer", name, tuple(b.shape), b.dtype)
            # hc_attn_fn_broadcast (a 2-D residual path) is not used by 3-D residual rows.
            process_weights_after_loading(layer, vc.model_config, DEV)
            layers.append(layer)
        prepare.declare_modules(*layers)
        for i, layer in enumerate(layers):
            # nxt=None: the stub attention owns no weights for the NEXT window, so
            # only the FFN window (mHC, ffn_norm, gate, shared expert) is planned.
            summary = layer.build_l2_prefetch(None)
            if verbose and i == 0:
                print("l2_prefetch", summary)
    return layers


class FFNInputs:
    """One decode step's FFN-sublayer inputs for M rows: the attention sublayer's
    output, the residual streams and mixes from the layer's mHC entry, input ids."""

    def __init__(self, layer, M: int, seed: int = 7):
        g = torch.Generator(device=DEV)
        g.manual_seed(seed + M)
        H = layer.hidden_size
        mult = layer.hc_mult
        residual0 = torch.randn((M, mult, H), generator=g, device=DEV).to(torch.bfloat16)
        residual, post_mix, res_mix, x, attn_pre, x_q = layer.mhc.pre(
            residual0,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.attn_norm.weight,
            None,
        )
        torch.cuda.synchronize()
        self.residual, self.post_mix, self.res_mix, self.attn_pre = residual, post_mix, res_mix, attn_pre
        self.attn_out = (torch.randn((M, H), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
        self.input_ids = torch.randint(3, 120000, (M,), generator=g, device=DEV, dtype=torch.int32)
        self.is_padding = torch.zeros((M,), dtype=torch.bool, device=DEV)
        self.M = M


def ffn_sublayer(layer, inp: FFNInputs):
    """The FFN half of DeepseekV4DecoderLayer.forward (sp_rows None)."""
    from vllm.models.deepseek_v4_1 import kernels

    residual, post_mix, res_mix, x, ffn_pre, x_q = layer.mhc.post_pre(
        inp.attn_out,
        inp.residual,
        inp.post_mix,
        inp.res_mix,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        layer.ffn_norm.weight,
        inp.attn_pre,
    )
    with kernels.mxfp8_inputs(x_q):
        x = layer.ffn(x, inp.input_ids)
    return x, residual, post_mix, res_mix, ffn_pre


def forward_context(vc, inp: FFNInputs):
    from vllm.forward_context import set_forward_context

    return set_forward_context(None, vc, num_tokens=inp.M, is_padding=inp.is_padding)
