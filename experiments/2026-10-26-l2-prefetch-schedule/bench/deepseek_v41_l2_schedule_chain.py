# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decode layers with their L2 prefetch windows, at one rank's TP4 or
TP3 shapes, for the window schedule's benchmark and test
(``vllm/models/deepseek_v4_1/l2_prefetch.py``).

``ScheduleChain(rows, tp)`` builds the target model's forty decoder layers in order,
each with its own weights: the two Engram layers (1 and 14), the four that hold a
compressor and an indexer (2, 8, 14 and 20: "own") and the four that hold only an
indexer (24, 28, 32 and 36: "idx"); the rest attend over the window and a reused
selection ("swa"). ``run(schedule)`` launches them in order on the current stream:

- real kernels for every projection a window prefetches, and for the weight streams
  that compete with a window for L2 and DRAM: the mHC entries (``post_pre``, as in
  ``deepseek_v41_pdl_chains``), fused Q-A/KV, Q-B, WO-A and WO-B, the router's BF16
  gate, the shared expert on its own stream beside the routed experts, the indexer's
  query and head-weight projections, the compressor's BF16 projection and the Engram
  projection with its input cast;
- spin kernels (``torch.cuda._sleep``: one thread, no memory traffic) for the
  latency-bound work around them: the attention core (RoPE, cache writes, metadata,
  the indexer's top-k, sparse MLA), the router's top-k, the attention and MoE
  all-reduces, and the Engram row wait and all-gathers. Each lasts rank 0's median in
  request-inventory-5's decode traces (TP4 ring, ports build; ``SPINS``).

A spin kernel neither reads memory nor slows under a fill, unlike sparse MLA (11.3 us
alone against 14.8 beside the WO fill on the ring) and the all-reduce (26-31 us with the
prefetch off against 39-41 at grid 2, 2026-10-06). Moving a fill off latency-bound work
gains more on the cluster than here.

The windows are ``l2_prefetch``'s: the plans ``DeepseekV4DecoderLayer.
build_l2_prefetch`` makes (same segments in the same order, same budgets), the
prefetcher's side stream and grid, one join at the end. A schedule says where each
window is issued (``SCHEDULES``). Every layer reads the same inputs, so the outputs
depend only on the kernels' arithmetic: the windows are cache hints, and every
schedule must give the same bits.
"""

import hashlib

import torch

from tests.kernels.deepseek_v41_pdl_chains import (
    DEVICE,
    EXPERTS,
    HIDDEN,
    LIMIT,
    Q_LATENT,
    QA_KV,
    TOPK,
    _block32,
    _Entry,
    _Projection,
    per_token_cast,
)
from vllm.models.deepseek_v4_1 import l2_prefetch
from vllm.models.deepseek_v4_1.tilelang import linear, mhc
from vllm.models.deepseek_v4_1.tilelang.activation import swiglu_clamp_kernel
from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes
from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm
from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection

LAYERS = 40
ENGRAM = (1, 14)
OWN = (2, 8, 14, 20)
IDX = (24, 28, 32, 36)
# One rank's shards: Q-B and WO-A over heads (16 heads in 2 groups at TP4, 24 in 3 at
# TP3), WO-B over K, the experts over their width. Q-A/KV, the indexer, the router,
# the compressor and the Engram projection are replicated.
SHAPES = {
    4: dict(
        q_b=(8192, Q_LATENT),
        groups=2,
        wo_a=(2048, 4096),
        wo_b=(HIDDEN, 2048),
        width=576,
    ),
    3: dict(
        q_b=(12288, Q_LATENT),
        groups=3,
        wo_a=(3072, 4096),
        wo_b=(HIDDEN, 3072),
        width=768,
    ),
}
INDEX_Q_B = (32 * 128, Q_LATENT)
INDEX_HEADS = 32
COMPRESSOR = 2 * 512  # wkv and wgate (ratio 2)
ENGRAM_WKV = (6400, 6144)
# Microseconds of latency-bound work per stretch: rank 0's medians in
# request-inventory-5's decode traces (TP4 ring; 6 rows: c1 greedy, 48: mixed-c8),
# less about 0.5 us a kernel the stretch launches (the chain's own launches add their
# gaps). 96 rows scale 48's attention core by 1.3 and its collectives by 1.15 (no
# trace). TP3 takes TP4's, with collectives at 0.7 (the triangle's one-shot
# collective, no relay).
SPINS = {
    6: dict(
        swa_core=41.5,
        idx_pre=10.5,
        idx_mid=7.0,
        idx_post=50.5,
        own_pre=8.5,
        own_mid=47.0,
        own_mid2=4.5,
        own_post=46.5,
        attn_ar=44.0,
        moe_ar=40.0,
        topk=5.0,
        eng_pre=57.5,
        eng_post=32.0,
    ),
    48: dict(
        swa_core=51.5,
        idx_pre=10.5,
        idx_mid=7.0,
        idx_post=69.5,
        own_pre=9.5,
        own_mid=46.0,
        own_mid2=4.5,
        own_post=72.5,
        attn_ar=58.0,
        moe_ar=51.5,
        topk=6.0,
        eng_pre=66.5,
        eng_post=48.0,
    ),
}
SPINS[96] = {k: v * (1.15 if k.endswith("_ar") else 1.3) for k, v in SPINS[48].items()}
# Distinct experts a layer's rows route to: about the routed GEMMs' traced times.
EXPERT_POOL = {6: 20, 48: 48, 96: 80}

# Where each window is issued:
# - head: the release (WO after Q-B, FFN after WO-B, NEXT after the combine);
# - chained: WO right after the fused Q-A/KV in layers without an indexer (its fill
#   follows NEXT's on the side stream, over the norms and the L2-fed Q-B), FFN right
#   after WO-A (over WO-B);
# - typed: chained, and the layers whose lines are lost before use get their own
#   points: an owner layer's WO after the indexer's head-weight projection, with an
#   IDX window (compressor, indexer Q-B and head weights) in WO's old place; before an
#   Engram layer an ENGRAM window (the Engram projection's first bytes) where NEXT
#   was, and NEXT right after the Engram projection;
# - off: no window (for reference).
SCHEDULES = ("head", "chained", "typed", "off")


class _BF16:
    """A BF16 projection as ``TileLangLinearMethod`` holds one."""

    def __init__(self, n: int, k: int, gen: torch.Generator, out_dtype=torch.bfloat16):
        self.weight = (
            torch.randn((n, k), generator=gen, device=DEVICE) * 0.05
        ).bfloat16()
        self.out_dtype = out_dtype
        linear.TileLangLinearMethod().declare_layer_kernels(self)

    def scratch(self, rows: int) -> torch.Tensor:
        nbytes = linear._spec_bytes(linear._partial_specs(self, rows))
        return torch.empty(max(nbytes, 256), dtype=torch.uint8, device=DEVICE)

    def __call__(self, x, out, scratch) -> None:
        linear._bf16_linear(x, out, self.tilelang_key, scratch)


class _Cast:
    """A block-32 projection's MXFP8 input buffers at ``rows``."""

    def __init__(self, rows: int, k: int, layer):
        self.rows = rows
        self.xq, self.sf = (
            torch.empty(shape, dtype=dtype, device=DEVICE)
            for shape, dtype in linear._activation_specs(
                rows, k, layer.tilelang_decode_rows
            )
        )

    def project(self, layer, x, out) -> None:
        per_token_cast(
            x,
            "e4m3",
            32,
            round_sf=True,
            use_packed_ue8m0=True,
            out=(self.xq[: self.rows], self.sf[: self.rows]),
        )
        linear._block32_gemm(layer, self.xq, self.sf, out, self.rows)


class _Layer:
    def __init__(
        self, index: int, rows: int, shapes: dict, gen: torch.Generator, shared
    ):
        def empty(cols, dtype=torch.bfloat16):
            return torch.empty((rows, cols), dtype=dtype, device=DEVICE)

        self.index, self.rows = index, rows
        self.kind = "own" if index in OWN else "idx" if index in IDX else "swa"
        self.engram = index in ENGRAM
        self.attention_entry = _Entry(rows, gen)
        self.ffn_entry = _Entry(rows, gen)
        self.qa_kv = _block32(*QA_KV, gen, mxfp8_input=True)
        self.q_b = _block32(*shapes["q_b"], gen)
        self.wo = TileLangWOProjection(
            _Projection(*shapes["wo_a"], gen),
            _Projection(*shapes["wo_b"], gen),
            groups=shapes["groups"],
        )
        self.wo.declare_kernels()
        width = shapes["width"]
        self.gate = _BF16(EXPERTS, HIDDEN, gen, out_dtype=torch.float32)
        self.gate_up = _block32(2 * width, HIDDEN, gen, mxfp8_input=True)
        self.down = _block32(HIDDEN, width, gen)
        self.casts = {
            "q_b": _Cast(rows, Q_LATENT, self.q_b),
            "down": _Cast(rows, width, self.down),
        }
        self.wo_xq, self.wo_sf = (
            torch.empty(shape, dtype=dtype, device=DEVICE)
            for shape, dtype in linear._activation_specs(
                rows, self.wo.a.columns, self.wo.b.tilelang_decode_rows
            )
        )
        self.partials = [
            torch.empty(shape, dtype=dtype, device=DEVICE)
            for shape, dtype in self.wo.a.partial_specs(rows)
        ]
        self.gate_scratch = self.gate.scratch(rows)
        self.qa, self.qb, self.wo_out = (
            empty(QA_KV[0]),
            empty(shapes["q_b"][0]),
            empty(HIDDEN),
        )
        self.qn, self.kvn = empty(Q_LATENT), empty(QA_KV[0] - Q_LATENT)
        self.logits = empty(EXPERTS, torch.float32)
        self.h, self.act = empty(2 * width), empty(width)
        self.shared_out, self.out = empty(HIDDEN), empty(HIDDEN)
        if self.kind in ("own", "idx"):
            self.index_q_b = _block32(*INDEX_Q_B, gen)
            self.casts["index_q_b"] = _Cast(rows, Q_LATENT, self.index_q_b)
            self.weights_proj = _BF16(INDEX_HEADS, HIDDEN, gen)
            self.wproj_scratch = self.weights_proj.scratch(rows)
            self.iq, self.iw = empty(INDEX_Q_B[0]), empty(INDEX_HEADS)
        if self.kind == "own":
            self.compressor = _BF16(COMPRESSOR, HIDDEN, gen, out_dtype=torch.float32)
            self.comp_scratch = self.compressor.scratch(rows)
            self.latent = empty(COMPRESSOR, torch.float32)
        if self.engram:
            self.wkv = _block32(*ENGRAM_WKV, gen)
            self.casts["wkv"] = _Cast(rows, ENGRAM_WKV[1], self.wkv)
            self.engram_rows = (
                torch.randn((rows, ENGRAM_WKV[1]), generator=gen, device=DEVICE) * 0.1
            ).bfloat16()
            self.kv = empty(ENGRAM_WKV[0])
        pool = torch.randperm(EXPERTS, generator=gen, device=DEVICE)[
            : EXPERT_POOL[rows]
        ]
        self.ids = torch.stack(
            [
                pool[torch.randperm(len(pool), generator=gen, device=DEVICE)[:TOPK]]
                for _ in range(rows)
            ]
        ).contiguous()
        self.route_weights = (
            torch.rand((rows, TOPK), generator=gen, device=DEVICE) + 0.1
        ).contiguous()
        self.swiglu, self.plan, self.experts, self.workspace = shared
        self.plans = {}

    def build_plans(self, nxt: "_Layer | None") -> None:
        """The windows ``build_l2_prefetch`` installs, and the typed schedule's own."""
        device = torch.device(DEVICE)
        f = self.ffn_entry

        def plan(segments, budget):
            return l2_prefetch.make_plan(segments, budget, device)

        def entry(name, e):
            return l2_prefetch.param_segments(
                "mhc", e, ("fn", "scale", "base")
            ) + l2_prefetch.param_segments(f"{name}_norm", e, ("weight",))

        self.plans["WO"] = plan(self.wo.l2_segments(), l2_prefetch.BUDGET_WO)
        self.plans["FFN"] = plan(
            entry("ffn", f)
            + l2_prefetch.linear_segments("gate", self.gate)
            + l2_prefetch.linear_segments("gate_up_proj", self.gate_up)
            + l2_prefetch.linear_segments("down_proj", self.down),
            l2_prefetch.BUDGET_FFN,
        )
        if nxt is not None:
            segments = entry("attn", nxt.attention_entry)
            segments += l2_prefetch.linear_segments("wqa_wkv", nxt.qa_kv)
            segments += l2_prefetch.linear_segments("wq_b", nxt.q_b)
            if nxt.kind in ("own", "idx"):
                segments += l2_prefetch.linear_segments("indexer.wq_b", nxt.index_q_b)
                segments += l2_prefetch.linear_segments(
                    "indexer.weights_proj", nxt.weights_proj
                )
            self.plans["NEXT"] = plan(segments, l2_prefetch.BUDGET_NEXT)
            if nxt.engram:
                self.plans["ENGRAM"] = plan(
                    l2_prefetch.linear_segments("engram.wkv", nxt.wkv),
                    l2_prefetch.BUDGET_NEXT,
                )
        if self.kind == "own":
            self.plans["IDX"] = plan(
                l2_prefetch.linear_segments("compressor", self.compressor)
                + l2_prefetch.linear_segments("indexer.wq_b", self.index_q_b)
                + l2_prefetch.linear_segments(
                    "indexer.weights_proj", self.weights_proj
                ),
                l2_prefetch.BUDGET_WO,
            )

    def run(self, schedule: str, inputs, previous: "_Layer | None", spin) -> None:
        rows, a, f, s = self.rows, self.attention_entry, self.ffn_entry, self.spins

        def issue(name):
            if schedule != "off":
                l2_prefetch.issue(self.plans.get(name), rows)

        typed = schedule == "typed"
        chained = schedule in ("chained", "typed")
        if self.engram:
            spin(s["eng_pre"])  # mHC post, row wait and gather, all-gather
            self.casts["wkv"].project(self.wkv, self.engram_rows, self.kv)
            if typed and previous is not None:
                l2_prefetch.issue(previous.plans.get("NEXT"), rows)
            spin(s["eng_post"])  # all-gather, gate
        a(*inputs)
        linear._block32_gemm(self.qa_kv, a.mx.values, a.mx.scales, self.qa, rows)
        if chained and self.kind == "swa":
            issue("WO")
        q_norm, kv_norm = self.norms
        qn = q_norm(self.qa[:, :Q_LATENT], out=self.qn)
        kv_norm(self.qa[:, Q_LATENT:], out=self.kvn)
        self.casts["q_b"].project(self.q_b, qn, self.qb)
        if typed and self.kind == "own":
            issue("IDX")
        elif not (chained and self.kind == "swa"):
            issue("WO")
        if self.kind == "swa":
            spin(s["swa_core"])
        else:
            if self.kind == "own":
                spin(s["own_pre"])  # RoPE, cache writes
                self.compressor(a.y, self.latent, self.comp_scratch)
                spin(s["own_mid"])  # compression, index keys, cache writes
            else:
                spin(s["idx_pre"])
            self.casts["index_q_b"].project(self.index_q_b, qn, self.iq)
            spin(s["idx_mid"] if self.kind == "idx" else s["own_mid2"])
            self.weights_proj(a.y, self.iw, self.wproj_scratch)
            if typed and self.kind == "own":
                issue("WO")
            spin(
                s["idx_post"] if self.kind == "idx" else s["own_post"]
            )  # top-k, sparse MLA
        self.wo.a(
            self.qb,
            self.wo_xq[:rows],
            self.wo_sf[:rows],
            self.partials[0] if self.partials else None,
        )
        if chained:
            issue("FFN")
        linear._block32_gemm(self.wo.b, self.wo_xq, self.wo_sf, self.wo_out, rows)
        if not chained:
            issue("FFN")
        spin(s["attn_ar"])
        f(self.wo_out, a.streams, a.post, a.comb, a.pre)
        main = torch.cuda.current_stream()
        aux = self.plan_aux
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            linear._block32_gemm(self.gate_up, f.mx.values, f.mx.scales, self.h, rows)
            self.swiglu(self.h, self.act)
            self.casts["down"].project(self.down, self.act, self.shared_out)
        self.gate(f.y, self.logits, self.gate_scratch)
        spin(s["topk"])

        def shared():
            main.wait_stream(aux)
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
        issue("ENGRAM" if typed and "ENGRAM" in self.plans else "NEXT")
        spin(s["moe_ar"])

    def outputs(self) -> dict[str, torch.Tensor]:
        out = {
            "q-a/kv": self.qa,
            "q-b": self.qb,
            "wo-b": self.wo_out,
            "logits": self.logits,
            "shared down": self.shared_out,
            "layer output": self.out,
        }
        for name, entry in (
            ("attention", self.attention_entry),
            ("ffn", self.ffn_entry),
        ):
            out[f"{name} y"] = entry.y
            out[f"{name} cast"] = entry.mx.values[: self.rows]
        if self.kind in ("own", "idx"):
            out["index q"], out["index weights"] = self.iq, self.iw
        if self.kind == "own":
            out["compressor"] = self.latent
        if self.engram:
            out["engram kv"] = self.kv
        return out


class ScheduleChain:
    """The forty layers at ``rows`` rows and one rank's ``tp`` shapes."""

    def __init__(self, rows: int, tp: int = 4, seed: int = 0, layers: int = LAYERS):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        shapes = SHAPES[tp]
        width = shapes["width"]
        self.rows = rows
        self.inputs = (
            (torch.randn((rows, HIDDEN), generator=gen, device=DEVICE)).bfloat16(),
            (
                torch.randn((rows, mhc.MULT * HIDDEN), generator=gen, device=DEVICE)
            ).bfloat16(),
            torch.rand((rows, mhc.MULT), generator=gen, device=DEVICE) * 2,
            torch.softmax(
                torch.randn((rows, mhc.MULT, mhc.MULT), generator=gen, device=DEVICE),
                -1,
            ),
            torch.softmax(
                torch.randn((rows, mhc.MULT), generator=gen, device=DEVICE), -1
            )
            + 1e-6,
        )
        plan = MoeExpertsV41Plan(
            EXPERTS, TOPK, HIDDEN, width, swiglu_limit=LIMIT, out_dtype=torch.bfloat16
        )
        plan.declare_kernels()

        def codes(*shape):
            return torch.empty(shape, dtype=torch.uint8, device=DEVICE).random_(
                generator=gen
            )

        def exponents(*shape):
            return torch.randint(
                118, 122, shape, generator=gen, device=DEVICE, dtype=torch.uint8
            )

        experts = plan.prepare_weights(
            codes(EXPERTS, 2 * width, HIDDEN // 2),
            exponents(EXPERTS, 2 * width, HIDDEN // 32),
            codes(EXPERTS, HIDDEN, width // 2),
            exponents(EXPERTS, HIDDEN, width // 32),
        )
        workspace = torch.empty(
            workspace_bytes(plan.workspace_specs(rows, torch.int64)),
            dtype=torch.uint8,
            device=DEVICE,
        )
        shared = (swiglu_clamp_kernel(width, LIMIT), plan, experts, workspace)
        self.layers = [_Layer(i, rows, shapes, gen, shared) for i in range(layers)]
        self.aux = torch.cuda.Stream()
        norms = []
        for width in (Q_LATENT, QA_KV[0] - Q_LATENT):
            norm = TileLangRMSNorm(width)
            norm.weight.data = (
                torch.randn(width, generator=gen, device=DEVICE) * 0.1 + 1.0
            ).bfloat16()
            norm.declare_kernels()
            norms.append(norm)
        spins = {
            k: v * (0.7 if tp == 3 and k.endswith("_ar") else 1.0)
            for k, v in SPINS[rows].items()
        }
        for layer in self.layers:
            layer.plan_aux = self.aux
            layer.norms = norms
            layer.spins = spins
        for layer, nxt in zip(self.layers, self.layers[1:] + [None]):
            layer.build_plans(nxt)
        # The prefetch kernel compiles before any capture, as the model's
        # build_l2_prefetch_plans warms it.
        if not l2_prefetch.warmup():
            raise RuntimeError("the L2 prefetch is off: VLLM_DS41_L2_PREFETCH or SM121")
        self.cycles_per_us = _cycles_per_us()

    def spin(self, us: float) -> None:
        torch.cuda._sleep(int(us * self.cycles_per_us))

    def run(self, schedule: str) -> None:
        previous = None
        for layer in self.layers:
            layer.run(schedule, self.inputs, previous, self.spin)
            previous = layer
        l2_prefetch.join()

    def plan_summary(self) -> dict:
        """Each window's plan of the first layer that has it."""
        out = {}
        for layer in self.layers:
            for name, plan in layer.plans.items():
                if plan is not None and name not in out:
                    out[name] = plan.describe()
        return out

    def digests(self) -> dict[str, str]:
        return {
            f"{layer.index} {name}": hashlib.sha256(
                tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()[:16]
            for layer in self.layers
            for name, tensor in layer.outputs().items()
        }


def _cycles_per_us() -> float:
    """``torch.cuda._sleep``'s clock: cycles a microsecond, measured."""
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    rates = []
    for _ in range(5):
        start.record()
        torch.cuda._sleep(2_000_000)
        stop.record()
        torch.cuda.synchronize()
        rates.append(2_000_000 / (start.elapsed_time(stop) * 1000))
    return sorted(rates)[2]
