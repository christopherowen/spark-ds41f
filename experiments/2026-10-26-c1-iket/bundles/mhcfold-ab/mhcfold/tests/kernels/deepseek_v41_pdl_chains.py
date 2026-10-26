# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One DeepSeek-V4.1 decode layer's kernel chain at TP4's shapes, for the PDL test
and benchmark (``vllm/models/deepseek_v4_1/tilelang/pdl.py``).

``LayerChain(rows)`` builds the layer's kernels on random weights; ``run()``
launches them in order, each kernel reading what the one before it wrote:

- the attention entry: mHC ``post_pre`` (the previous FFN output folded into the
  streams) and its finalize, which also writes its output's MXFP8 cast (one launch
  at decode sizes, ``mhc.FOLD_ROWS``);
- the fused Q-A/KV projection's GEMM on that cast, the q latent's RMSNorm, Q-B
  (cast, GEMM);
- WO-A's split-K partials and reduce-cast over Q-B's output, which stands in for
  the sparse-MLA output of the same ``[rows, 16 x 512]`` shape, then WO-B;
- the FFN entry on WO-B's output;
- the shared expert on its own stream (gate/up on the FFN entry's cast, SwiGLU,
  cast, down) beside the routed experts (dispatch, FC1 on the same cast, SwiGLU and
  cast, FC2), and the combine, which waits for the shared expert and adds its output
  before one rounding.

That is the decode layer without its attention core (RoPE, cache writes, sparse-MLA
metadata and attention), its router and its all-reduces: 20 kernels, 18 with the
entries folded. Expert routes are fixed by the seed. The same seed gives the same
inputs, weights and routes, so the outputs depend only on the kernels' arithmetic,
whatever build or launch schedule runs them. Only APIs that predate PDL are used,
so the chain runs on the build without it too.
"""

import hashlib

import torch

from vllm.models.deepseek_v4_1.tilelang import linear, mhc

try:  # a build with PDL casts the activations as its projections do, chained
    from vllm.models.deepseek_v4_1.tilelang.pdl import per_token_cast
except ImportError:  # the build before PDL
    import tile_kernels

    per_token_cast = tile_kernels.quant.per_token_cast
from vllm.models.deepseek_v4_1.tilelang.activation import swiglu_clamp_kernel
from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes
from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm
from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection

DEVICE = "cuda"
HIDDEN = 5120
# TP4 serving shapes (gemm.DECODE_FP8_CONFIGS): fused Q-A/KV, Q-B, grouped WO-A,
# WO-B, the shared expert's gate/up and down.
QA_KV = (1792, HIDDEN)
Q_LATENT = 1280
Q_B = (8192, Q_LATENT)
WO_GROUPS = 2
WO_A = (2048, 4096)
WO_B = (HIDDEN, 2048)
SHARED = 576
EXPERTS, TOPK = 384, 6
LIMIT = 10.0
RMS_EPS, HC_EPS, SINKHORN = 1e-20, 1e-6, 20
# A decode step's rows: one stream (1 + 5 drafted), eight mixed streams, eight at
# six rows each.
ROWS = {"c1": 6, "c8": 24, "c8x6": 48}


class _Projection:
    """A block-32 FP8 weight as the TileLang linear method holds one."""

    def __init__(self, n: int, k: int, gen: torch.Generator):
        self.weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(
            torch.float8_e4m3fn
        )
        self.weight_scale_inv = torch.randint(
            120,
            125,
            (n // 32, k // 32),
            generator=gen,
            device=DEVICE,
            dtype=torch.uint8,
        )


def _block32(
    n: int, k: int, gen: torch.Generator, mxfp8_input: bool = False
) -> _Projection:
    """``mxfp8_input``: the projection reads the cast its input's producer wrote (an
    mHC entry's), as the model's fused Q-A/KV and shared gate/up do."""
    layer = _Projection(n, k, gen)
    layer.tilelang_mxfp8_input = mxfp8_input
    linear._block32_scale_words(layer)
    linear._declare_block32(layer)
    return layer


class _Entry:
    """An mHC sublayer entry (``post_pre``): the streams, the incoming mixes and fn;
    its output ``y`` and that output's MXFP8 cast ``mx``."""

    def __init__(self, rows: int, gen: torch.Generator):
        def randn(*shape, scale=1.0):
            return torch.randn(*shape, generator=gen, device=DEVICE) * scale

        splits = HIDDEN // mhc.BLOCK_H
        self.fn = randn(mhc.MIXES, mhc.MULT * HIDDEN, scale=0.02)
        self.scale = randn(3, scale=0.1) + 1.0
        self.base = randn(mhc.MIXES, scale=0.1)
        self.weight = (randn(HIDDEN, scale=0.1) + 1.0).bfloat16()
        self.streams = torch.empty(
            (rows, mhc.MULT * HIDDEN), dtype=torch.bfloat16, device=DEVICE
        )
        self.P = torch.empty((splits, rows, 1, mhc.MIXES), device=DEVICE)
        self.S = torch.empty((splits, rows, 1), device=DEVICE)
        self.C = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
        self.Q = torch.empty((splits, rows), device=DEVICE)
        self.post = torch.empty((rows, mhc.MULT), device=DEVICE)
        self.comb = torch.empty((rows, mhc.MULT, mhc.MULT), device=DEVICE)
        self.pre = torch.empty((rows, mhc.MULT), device=DEVICE)
        self.y = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
        self.mx = linear.empty_mxfp8(self.y)
        self.project = mhc.project_streams(
            HIDDEN, "post_pre", block_M=mhc.BLOCK_M, world=1
        )
        self.finalize = mhc.finalize(
            HIDDEN, mhc.MULT * HIDDEN, RMS_EPS, HC_EPS, SINKHORN, RMS_EPS
        )
        # At decode sizes the model launches the entry once, the finalize folded in.
        self.fold = (
            mhc.project_streams(
                HIDDEN,
                "post_pre",
                block_M=mhc.BLOCK_M,
                fold=(RMS_EPS, HC_EPS, SINKHORN, RMS_EPS),
            )
            if rows <= mhc.FOLD_ROWS
            else None
        )
        self.barrier = torch.zeros(2, dtype=torch.int32, device=DEVICE)

    def __call__(self, x, streams, post, comb, pre) -> None:
        """Fold ``x`` (post, comb) into ``streams``; project, collapse with ``pre``."""
        projection = (
            x.unsqueeze(0),
            streams,
            post,
            comb,
            pre,
            self.fn,
            self.streams,
            self.P,
            self.S,
            self.C,
            self.Q,
        )
        if self.fold is not None:
            self.fold(
                *projection,
                self.scale,
                self.base,
                self.weight,
                self.post,
                self.comb,
                self.pre,
                self.y,
                self.mx.values,
                self.mx.scales,
                self.barrier,
            )
            return
        self.project(*projection)
        self.finalize(
            self.P,
            self.S,
            self.C,
            self.Q,
            self.scale,
            self.base,
            self.weight,
            self.post,
            self.comb,
            self.pre,
            self.y,
            self.mx.values,
            self.mx.scales,
        )


class LayerChain:
    """The layer's kernels and buffers at ``rows`` rows; see the module docstring."""

    def __init__(self, rows: int, seed: int = 0):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)

        def randn(*shape, scale=1.0):
            return torch.randn(*shape, generator=gen, device=DEVICE) * scale

        self.rows = rows
        # The chain's inputs: the streams, the previous FFN output and its mixes.
        self.x = randn(rows, HIDDEN).bfloat16()
        self.streams = randn(rows, mhc.MULT * HIDDEN).bfloat16()
        self.post_in = torch.rand((rows, mhc.MULT), generator=gen, device=DEVICE) * 2
        self.comb_in = torch.softmax(randn(rows, mhc.MULT, mhc.MULT), dim=-1)
        self.pre_in = torch.softmax(randn(rows, mhc.MULT), dim=-1) + 1e-6
        self.attention_entry = _Entry(rows, gen)
        self.ffn_entry = _Entry(rows, gen)
        self.qa_kv = _block32(*QA_KV, gen, mxfp8_input=True)
        self.q_norm = TileLangRMSNorm(Q_LATENT)
        self.q_norm.weight.data = (randn(Q_LATENT, scale=0.1) + 1.0).bfloat16()
        self.q_norm.declare_kernels()
        self.q_b = _block32(*Q_B, gen)
        self.wo = TileLangWOProjection(
            _Projection(*WO_A, gen), _Projection(*WO_B, gen), groups=WO_GROUPS
        )
        self.wo.declare_kernels()
        self.gate_up = _block32(2 * SHARED, HIDDEN, gen, mxfp8_input=True)
        self.down = _block32(HIDDEN, SHARED, gen)
        self.swiglu = swiglu_clamp_kernel(SHARED, LIMIT)
        self.plan = MoeExpertsV41Plan(
            EXPERTS,
            TOPK,
            HIDDEN,
            SHARED,
            swiglu_limit=LIMIT,
            out_dtype=torch.float32,
        )
        self.plan.declare_kernels()

        def codes(*shape):  # packed E2M1 pairs
            out = torch.empty(shape, dtype=torch.uint8, device=DEVICE)
            return out.random_(generator=gen)

        def exponents(*shape):  # UE8M0 scales 2^-9 to 2^-6
            return torch.randint(
                118, 122, shape, generator=gen, device=DEVICE, dtype=torch.uint8
            )

        w13 = codes(EXPERTS, 2 * SHARED, HIDDEN // 2)
        w13_sf = exponents(EXPERTS, 2 * SHARED, HIDDEN // 32)
        w2 = codes(EXPERTS, HIDDEN, SHARED // 2)
        w2_sf = exponents(EXPERTS, HIDDEN, SHARED // 32)
        self.experts = self.plan.prepare_weights(w13, w13_sf, w2, w2_sf)
        del w13, w13_sf, w2, w2_sf
        self.ids = torch.stack(
            [
                torch.randperm(EXPERTS, generator=gen, device=DEVICE)[:TOPK]
                for _ in range(rows)
            ]
        ).contiguous()
        self.route_weights = (
            torch.rand((rows, TOPK), generator=gen, device=DEVICE) + 0.1
        ).contiguous()
        self.workspace = torch.empty(
            workspace_bytes(self.plan.workspace_specs(rows, self.ids.dtype)),
            dtype=torch.uint8,
            device=DEVICE,
        )

        def empty(cols, dtype=torch.bfloat16):
            return torch.empty((rows, cols), dtype=dtype, device=DEVICE)

        self.qa = empty(QA_KV[0])
        self.qn = empty(Q_LATENT)
        self.qb = empty(Q_B[0])
        self.wo_out = empty(HIDDEN)
        self.h = empty(2 * SHARED)
        self.act = empty(SHARED)
        self.shared_out = empty(HIDDEN)
        self.out = empty(HIDDEN)
        self.casts = {
            name: [
                torch.empty(shape, dtype=dtype, device=DEVICE)
                for shape, dtype in linear._activation_specs(
                    rows, k, layer.tilelang_decode_rows
                )
            ]
            for name, k, layer in (
                ("q_b", Q_LATENT, self.q_b),
                ("wo_b", WO_B[1], self.wo.b),
                ("down", SHARED, self.down),
            )
        }
        self.partials = [
            torch.empty(shape, dtype=dtype, device=DEVICE)
            for shape, dtype in self.wo.a.partial_specs(rows)
        ]
        self.aux = torch.cuda.Stream()

    def _project(self, name: str, layer: _Projection, x, out) -> None:
        """A block-32 projection: TileKernels' MXFP8 cast, then the decode GEMM."""
        xq, sf = self.casts[name]
        per_token_cast(
            x,
            "e4m3",
            32,
            round_sf=True,
            use_packed_ue8m0=True,
            out=(xq[: self.rows], sf[: self.rows]),
        )
        linear._block32_gemm(layer, xq, sf, out, self.rows)

    def run(self) -> None:
        """The layer's kernels on the current stream (the shared expert on ``aux``)."""
        rows, a, f = self.rows, self.attention_entry, self.ffn_entry
        a(self.x, self.streams, self.post_in, self.comb_in, self.pre_in)
        linear._block32_gemm(self.qa_kv, a.mx.values, a.mx.scales, self.qa, rows)
        self.q_norm(self.qa[:, :Q_LATENT], out=self.qn)
        self._project("q_b", self.q_b, self.qn, self.qb)
        xq, sf = self.casts["wo_b"]
        self.wo.a(
            self.qb, xq[:rows], sf[:rows], self.partials[0] if self.partials else None
        )
        linear._block32_gemm(self.wo.b, xq, sf, self.wo_out, rows)
        f(self.wo_out, a.streams, a.post, a.comb, a.pre)
        main = torch.cuda.current_stream()
        self.aux.wait_stream(main)
        with torch.cuda.stream(self.aux):
            linear._block32_gemm(self.gate_up, f.mx.values, f.mx.scales, self.h, rows)
            self.swiglu(self.h, self.act)
            self._project("down", self.down, self.act, self.shared_out)

        def shared():
            main.wait_stream(self.aux)
            return self.shared_out

        self.plan.run(
            f.mx,
            self.ids,
            self.route_weights,
            self.experts,
            self.out,
            self.workspace,
            shared=shared,
        )

    def outputs(self) -> dict[str, torch.Tensor]:
        """Every buffer a kernel of the chain writes (live rows), the layer's output
        last. The routed experts' workspace is left out: the dispatch owns it."""
        a, f, rows = self.attention_entry, self.ffn_entry, self.rows
        out = {}
        for name, entry in (("attention", a), ("ffn", f)):
            for field in ("streams", "P", "S", "C", "Q", "post", "comb", "pre", "y"):
                out[f"{name} {field}"] = getattr(entry, field)
            out[f"{name} cast"] = entry.mx.values[:rows]
            out[f"{name} scales"] = entry.mx.scales[:rows]
        for name, (xq, sf) in self.casts.items():
            out[f"{name} cast"] = xq[:rows]
            out[f"{name} scales"] = sf[:rows]
        for i, partials in enumerate(self.partials):
            out[f"wo-a partials {i}"] = partials
        out.update(
            {
                "q-a/kv": self.qa,
                "q latent": self.qn,
                "q-b": self.qb,
                "wo-b": self.wo_out,
                "shared gate/up": self.h,
                "shared swiglu": self.act,
                "shared down": self.shared_out,
                "layer output": self.out,
            }
        )
        return out

    def poison(self) -> None:
        """NaN in every buffer ``outputs`` lists, padding rows included: a kernel that
        read one before its producer wrote it would carry a NaN to the output."""
        for tensor in self.outputs().values():
            if tensor.dtype.is_floating_point and tensor.element_size() > 1:
                tensor.fill_(float("nan"))
        casts = [*self.casts.values()]
        casts += [entry.mx[1:] for entry in (self.attention_entry, self.ffn_entry)]
        for xq, sf in casts:
            xq.view(torch.uint8).fill_(0x7F)  # E4M3 NaN
            sf.fill_(0xFF)  # UE8M0 NaN

    def digests(self) -> dict[str, str]:
        """A short SHA-256 of each buffer ``outputs`` lists."""
        return {
            name: hashlib.sha256(
                tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()[:16]
            for name, tensor in self.outputs().items()
        }
