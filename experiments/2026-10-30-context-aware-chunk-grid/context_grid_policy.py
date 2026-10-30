"""Context-aware step-time budgeting on an absolute 2,048-token grid.

Models and evaluates `Scheduler._context_aware_chunk_threshold` and
`Scheduler._aligned_prefill_chunk` when `--prefill-step-target-s` (tau_max)
is enabled:

- When `num_runnable_decodes == 0` (solo prefill): uses the full
  `long_prefill_token_threshold` (16,288 on TP4, 6,096 / 8,144 on TP3).
  Solo 32K / 131K / 1M prefill speed is 100% unchanged.
- When `num_runnable_decodes > 0` (mixed prefill + decode): chooses the
  chunk's target threshold `T(c) in {2048, 4096, 8192, threshold}` on the
  absolute 2,048-token grid to keep the step duration within `tau_max`
  (default 1.0 s):
      T(c) = quantize_2048((tau_max - F - alpha * t_dec) / (m0 + m1 * c))

Standard library only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STALL_DIR = ROOT / "experiments/2026-10-26-concurrency-stalls"
if str(STALL_DIR) not in sys.path:
    sys.path.insert(0, str(STALL_DIR))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402
import stallsim  # noqa: E402
import workloads  # noqa: E402
from stallsim import Engine, Lineage, Policy, Shape, percentile, summarize  # noqa: E402

EXPERIMENT = "experiments/2026-10-30-context-aware-chunk-grid"
IMAGE_VLLM = "/opt/spark-ds41f/candidate/vllm"
OVERLAY = (
    "vllm/v1/core/sched/scheduler.py",
    "vllm/config/scheduler.py",
    "vllm/engine/arg_utils.py",
)
RECIPE = "releases/spark-ds41f-1025/cluster.json"

# Calibrated TP4 step-time parameters (calibrate.py / calibration.json):
TP4_F_S = 0.1535
TP4_ALPHA = 1.56
TP4_M0_S_PER_TOK = 1.2532308400535526e-4
TP4_M1_S_PER_TOK2 = 1.3289467958204244e-10
TP4_DEC_KNOTS: tuple[tuple[int, float], ...] = (
    (1, 0.02983160068849846),
    (8, 0.08379558412906112),
    (16, 0.11908994468658608),
)


def estimate_t_dec(
    num_runnable_decodes: int,
    knots: tuple[tuple[int, float], ...] = TP4_DEC_KNOTS,
) -> float:
    if num_runnable_decodes <= 0:
        return 0.0
    if num_runnable_decodes <= knots[0][0]:
        return knots[0][1]
    for (n0, t0), (n1, t1) in zip(knots, knots[1:]):
        if num_runnable_decodes <= n1:
            return t0 + (t1 - t0) * (num_runnable_decodes - n0) / (n1 - n0)
    (n0, t0), (n1, t1) = knots[-2], knots[-1]
    return t1 + (num_runnable_decodes - n1) * (t1 - t0) / (n1 - n0)


def quantize_chunk_threshold(
    computed: int,
    num_runnable_decodes: int,
    *,
    threshold: int = 16288,
    step_target_s: float = 1.0,
    grid: int = 2048,
    F: float = TP4_F_S,
    alpha: float = TP4_ALPHA,
    m0: float = TP4_M0_S_PER_TOK,
    m1: float = TP4_M1_S_PER_TOK2,
    dec_knots: tuple[tuple[int, float], ...] = TP4_DEC_KNOTS,
) -> int:
    """Quantize the step-time budget onto `{2048, 4096, 8192, threshold}`."""
    if threshold <= 0 or num_runnable_decodes <= 0 or step_target_s <= 0.0:
        return threshold
    grid = max(int(grid), 1)
    t_dec = estimate_t_dec(num_runnable_decodes, knots=dec_knots)
    avail_s = max(step_target_s - F - alpha * t_dec, 0.0)
    denom = m0 + m1 * max(computed, 0)
    q_raw = avail_s / denom if denom > 0.0 else float(threshold)
    grid_units = max(1, int((q_raw + 0.75 * grid) // grid))
    if grid_units <= 1:
        tier = grid
    elif grid_units == 2:
        tier = 2 * grid
    elif grid_units <= 4:
        tier = 4 * grid
        if tier > threshold >= 2 * grid:
            tier = 2 * grid
    else:
        tier = threshold
    return min(tier, threshold)


def aligned_prefill_chunk(
    computed: int,
    remaining: int,
    *,
    threshold: int = 16288,
    num_runnable_decodes: int = 0,
    step_target_s: float = 1.0,
    grid: int = 2048,
    F: float = TP4_F_S,
    alpha: float = TP4_ALPHA,
    m0: float = TP4_M0_S_PER_TOK,
    m1: float = TP4_M1_S_PER_TOK2,
    dec_knots: tuple[tuple[int, float], ...] = TP4_DEC_KNOTS,
) -> int:
    """Return the next aligned prefill chunk size."""
    if threshold <= 0:
        return remaining
    tier = quantize_chunk_threshold(
        computed,
        num_runnable_decodes,
        threshold=threshold,
        step_target_s=step_target_s,
        grid=grid,
        F=F,
        alpha=alpha,
        m0=m0,
        m1=m1,
        dec_knots=dec_knots,
    )
    if tier >= threshold:
        return min(remaining, threshold - computed % threshold)
    grid = max(int(grid), 1)
    end = ((computed + tier) // grid) * grid
    if end <= computed:
        end = ((computed // grid) + 1) * grid
    return min(remaining, end - computed)


def chunk_step_duration_s(
    chunk_tokens: int,
    computed: int,
    num_runnable_decodes: int,
    *,
    F: float = TP4_F_S,
    alpha: float = TP4_ALPHA,
    m0: float = TP4_M0_S_PER_TOK,
    m1: float = TP4_M1_S_PER_TOK2,
    dec_knots: tuple[tuple[int, float], ...] = TP4_DEC_KNOTS,
) -> float:
    """Step duration `F + alpha * t_dec + m0 * q + m1 * q * (c + q / 2)`."""
    t_dec = estimate_t_dec(num_runnable_decodes, knots=dec_knots)
    c_mid = computed + 0.5 * chunk_tokens
    return F + alpha * t_dec + m0 * chunk_tokens + m1 * chunk_tokens * c_mid


def load_shapes() -> dict[str, Shape]:
    raw = json.loads((STALL_DIR / "calibration.json").read_text())
    tp4 = Shape(**raw["tp4-ring"])
    tp3 = Shape(**raw["tp3-triangle"])
    tp3_6k = dataclasses.replace(tp3, name="tp3-6k", budget=6144, fill_blocks=1429)
    return {"tp4": tp4, "tp3": tp3, "tp3_6k": tp3_6k}


def evaluate_solo_prefills(shapes: dict[str, Shape]) -> dict:
    """Verify solo prefill (num_runnable_decodes == 0) is 100% unchanged."""
    out: dict = {}
    for shape_key, prompt_sizes in (
        ("tp4", (32768, 131072, 1048576)),
        ("tp3_6k", (32768, 131072)),
    ):
        shape = shapes[shape_key]
        shape_out: dict = {}
        for prompt in prompt_sizes:
            base_engine = Engine(shape, Policy(name="production"), seed=1)
            base_req = base_engine.submit("solo", 0.0, prompt, 1, Lineage())
            base_engine.run(2000.0)

            grid_engine = Engine(
                shape,
                Policy(name="context-grid", step_target_s=1.0, grid=2048),
                seed=1,
            )
            grid_req = grid_engine.submit("solo", 0.0, prompt, 1, Lineage())
            grid_engine.run(2000.0)

            base_ttft = base_req.first - base_req.sent
            grid_ttft = grid_req.first - grid_req.sent
            base_chunks = [s.prefill_tokens for s in base_engine.steps if s.prefill_tokens > 0]
            grid_chunks = [s.prefill_tokens for s in grid_engine.steps if s.prefill_tokens > 0]
            shape_out[f"{prompt // 1024}k"] = {
                "prompt_tokens": prompt,
                "threshold": shape.threshold,
                "production_chunks": base_chunks,
                "context_grid_chunks": grid_chunks,
                "production_ttft_s": round(base_ttft, 6),
                "context_grid_ttft_s": round(grid_ttft, 6),
                "ttft_delta_pct": round((grid_ttft - base_ttft) / base_ttft * 100.0, 6),
            }
        out[shape_key] = shape_out
    return out


def evaluate_context_sweep() -> list[dict]:
    """Compare static 16,288-token chunks vs context-aware T(c) across context depth."""
    rows: list[dict] = []
    n_dec = 1
    for c in (0, 32768, 65536, 131072, 262144, 524288, 900000):
        static_q = 16288
        static_step_s = chunk_step_duration_s(static_q, c, n_dec)
        tier_q = quantize_chunk_threshold(c, n_dec, threshold=16288, step_target_s=1.0, grid=2048)
        tier_step_s = chunk_step_duration_s(tier_q, c, n_dec)
        # Prefill time to advance 16,288 tokens at context c using tier_q sub-chunks:
        sub_chunks = static_q / tier_q
        equiv_prefill_s = sub_chunks * tier_step_s
        rows.append(
            {
                "context_tokens": c,
                "static_chunk_tokens": static_q,
                "static_step_s": round(static_step_s, 4),
                "context_grid_tier_tokens": tier_q,
                "context_grid_step_s": round(tier_step_s, 4),
                "step_stall_reduction_pct": round(
                    (1.0 - tier_step_s / static_step_s) * 100.0, 2
                ),
                "prefill_time_overhead_pct": round(
                    (equiv_prefill_s / static_step_s - 1.0) * 100.0, 2
                ),
            }
        )
    return rows


def run_stall_case(
    shape: Shape,
    policy: Policy,
    *,
    streams: int,
    doc_tokens: int,
    probes: tuple[float, ...] = (0.5, 2.0, 5.0),
) -> dict:
    engine = Engine(shape, policy, seed=1)
    info = workloads.stall_probe(engine, streams, doc_tokens, warmup=20.0, probes=probes)
    doc = info["doc"]
    engine.run(600.0)
    t0 = doc.admitted if doc.admitted is not None else 20.0
    t1 = doc.prefilled if doc.prefilled is not None else engine.now
    span = max(t1 - t0, 1e-6)
    stream_reqs = [r for r in engine.requests if r.cls == "stream"]
    probe_reqs = info["probes"]

    # Stream gaps that overlap the document's prefill window [t0, t1]:
    window_gaps: list[float] = []
    for r in stream_reqs:
        t_emit = r.first
        if t_emit is None:
            continue
        for g in r.gaps:
            prev_t = t_emit
            t_emit = prev_t + g
            if t_emit >= t0 and prev_t <= t1:
                window_gaps.append(g)

    steps_in_window = [s for s in engine.steps if t0 - 1e-9 <= s.start < t1 + 1e-9]
    chunk_steps = [s for s in steps_in_window if s.prefill_tokens >= 1024]
    decode_steps = [s for s in steps_in_window if s.prefill_tokens < 1024 and s.n_dec > 0]
    emitted_in_window = sum(s.emitted for s in steps_in_window)
    probe_ttfts = [p.first - p.sent for p in probe_reqs if p.first is not None]

    return {
        "doc_prefill_s": round(span, 4),
        "doc_ttft_s": round(doc.first - doc.sent, 4) if doc.first is not None else None,
        "chunk_steps": len(chunk_steps),
        "interleaved_decode_steps": len(decode_steps),
        "max_chunk_step_s": round(max((s.duration for s in chunk_steps), default=0.0), 4),
        "p99_step_s": round(percentile([s.duration for s in steps_in_window], 0.99) or 0.0, 4),
        "stream_itl_mean_s": round(sum(window_gaps) / len(window_gaps), 4) if window_gaps else None,
        "stream_itl_p50_s": round(percentile(window_gaps, 0.5) or 0.0, 4),
        "stream_itl_p99_s": round(percentile(window_gaps, 0.99) or 0.0, 4),
        "stream_itl_max_s": round(max(window_gaps), 4) if window_gaps else None,
        "decode_tok_s_during_prefill": round(emitted_in_window / span, 2),
        "probe_ttft_mean_s": round(sum(probe_ttfts) / len(probe_ttfts), 4) if probe_ttfts else None,
        "probe_ttft_max_s": round(max(probe_ttfts), 4) if probe_ttfts else None,
    }


def run_soak_case(shape: Shape, policy: Policy) -> dict:
    """Deep-context multi-session soak (sessions growing to 960K context) plus background chat."""
    engine = Engine(shape, policy, seed=7)
    rng = random.Random(107)
    workloads.soak_agents(engine, rng, sessions=3, start=0.0, end=360.0)
    workloads.closed_loop(
        engine,
        random.Random(207),
        users=4,
        start=0.0,
        end=360.0,
        prompt=lambda r: 64,
        out=lambda r: 256,
        think=lambda r: 0.2,
        cls="chat",
        stagger=0.5,
    )
    engine.run(420.0)
    summary = summarize(engine, 60.0, 360.0)
    chat = summary["classes"].get("chat", {})
    soak = summary["classes"].get("soak", {})
    window_steps = [s for s in engine.steps if 60.0 <= s.start < 360.0]
    chunk_steps = [s for s in window_steps if s.prefill_tokens >= 1024]
    return {
        "max_chunk_step_s": round(max((s.duration for s in chunk_steps), default=0.0), 4),
        "step_p99_s": round(summary["step_p99"] or 0.0, 4),
        "chat_ttft_p50_s": round(chat.get("ttft_p50") or 0.0, 4),
        "chat_ttft_p99_s": round(chat.get("ttft_p99") or 0.0, 4),
        "chat_itl_p99_s": round(chat.get("itl_p99") or 0.0, 4),
        "chat_itl_max_s": round(chat.get("itl_max") or 0.0, 4),
        "soak_ttft_p50_s": round(soak.get("ttft_p50") or 0.0, 4),
        "output_tok_s": round(summary["output_tok_s"], 2),
        "prefill_tok_s": round(summary["prefill_tok_s"], 2),
    }


def evaluate_all() -> dict:
    shapes = load_shapes()
    policies = {
        "production": Policy(name="production"),
        "interleave_k2": Policy(
            name="interleave-k2",
            order="bypass",
            first_token_priority=True,
            long_wait_cap_s=10.0,
            interleave_decode_steps=2,
        ),
        "context_grid": Policy(
            name="context-grid",
            step_target_s=1.0,
            grid=2048,
        ),
        "context_grid_interleave_k2": Policy(
            name="context-grid+interleave-k2",
            order="bypass",
            first_token_priority=True,
            long_wait_cap_s=10.0,
            step_target_s=1.0,
            grid=2048,
            interleave_decode_steps=2,
        ),
    }

    stall_specs = {
        "stall-15": ("tp4", 15, 65536, (0.5, 2.0, 5.0)),
        "stall-11": ("tp4", 11, 65536, (0.5, 2.0, 5.0)),
        "stall-7-tp3": ("tp3", 7, 32768, (0.5, 2.0, 4.0)),
        "stall-3-tp3": ("tp3_6k", 3, 32768, (0.5, 2.0, 4.0)),
        "stall-1m": ("tp4", 1, 900000, (1.0, 10.0, 30.0)),
    }

    workloads_out: dict = {}
    for name, (shape_key, streams, doc_tokens, probes) in stall_specs.items():
        shape = shapes[shape_key]
        workloads_out[name] = {
            arm_key: run_stall_case(
                shape,
                policy,
                streams=streams,
                doc_tokens=doc_tokens,
                probes=probes,
            )
            for arm_key, policy in policies.items()
        }

    workloads_out["soak"] = {
        arm_key: run_soak_case(shapes["tp4"], policy)
        for arm_key, policy in policies.items()
    }

    return {
        "solo_prefills": evaluate_solo_prefills(shapes),
        "context_sweep_tp4": evaluate_context_sweep(),
        "workloads": workloads_out,
    }


def make_arm(
    shape: str,
    *,
    step_target_s: float = 1.0,
    grid: int = 2048,
    interleave_steps: int = 0,
) -> dict:
    cluster = json.loads((ROOT / RECIPE).read_text())
    resolved = recipe.resolve(cluster, shape)
    mounts = [
        m for m in copy.deepcopy(resolved["container"]["mounts"]) if m[1] == "/cache"
    ]
    mounts += [
        [f"{{checkout}}/{EXPERIMENT}/overlay/vllm/{path}", f"{IMAGE_VLLM}/{path}", "ro"]
        for path in OVERLAY
    ]
    extra_args = [
        "--prefill-step-target-s",
        str(step_target_s),
        "--prefill-chunk-grid",
        str(grid),
    ]
    if interleave_steps > 0:
        extra_args = [
            "--prefill-yield",
            "short",
            "--prefill-yield-limit-s",
            "10",
            "--prefill-decode-interleave-steps",
            str(interleave_steps),
        ] + extra_args
    arm = {"extends": {"config": RECIPE, "shape": shape}}
    arm["notes"] = (
        f"spark-ds41f-1025 with context-aware step-time budgeting on the {grid}-token "
        f"grid (tau_max={step_target_s}s, K={interleave_steps} interleaved decode steps)."
    )
    arm["set"] = {
        "serve_args": resolved["serve_args"] + extra_args,
        "container": {"mounts": mounts},
    }
    return arm


def write_artifacts() -> dict:
    results = evaluate_all()
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    arms = {
        "context-grid-tp4.json": make_arm(
            "tp4-ring", step_target_s=1.0, grid=2048, interleave_steps=0
        ),
        "context-grid-interleave-k2-tp4.json": make_arm(
            "tp4-ring", step_target_s=1.0, grid=2048, interleave_steps=2
        ),
        "context-grid-interleave-k2-tp3.json": make_arm(
            "tp3-triangle", step_target_s=1.0, grid=2048, interleave_steps=2
        ),
    }
    for filename, payload in arms.items():
        (HERE / filename).write_text(json.dumps(payload, indent=2) + "\n")
    return results


if __name__ == "__main__":
    write_artifacts()
