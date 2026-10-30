#!/usr/bin/env python3
"""Evaluate K-step CUDA-graphed decode interleaving between full prefill chunks.

Compares four scheduling policies on the calibrated spark-ds41f shapes (tp4-ring
and tp3-triangle):
  1. production: FCFS chunked prefill (K=0, prefill_yield="off").
  2. prefill-yield-short: --prefill-yield short --prefill-yield-limit-s 10.
  3. interleave-k1: --prefill-decode-interleave-steps 1 (K=1 graphed decode step
     between full prefill chunks, admitting waiting short prefills).
  4. interleave-k2: --prefill-decode-interleave-steps 2 (K=2 graphed decode steps
     between full prefill chunks, admitting waiting short prefills).

Workloads evaluated:
  - stall-15 (tp4-ring): 15 active decoding streams + 131K prefill (100% slot
    occupancy, 16/16 seqs; --prefill-yield short cannot trigger because
    admission_open is False).
  - stall-11 (tp4-ring): 11 active decoding streams + 131K prefill + 5 arriving
    500-token short probes during the prefill.
  - stall-7 (tp3-triangle): 7 active decoding streams + 131K prefill (100% slot
    occupancy, 8/8 seqs on the 3-node triangle).
  - stall-3 (tp3-triangle): 3 active decoding streams + 131K prefill + 5 arriving
    500-token short probes during the prefill.
  - stall-1m (tp4-ring): 11 active decoding streams + 1,000,000-token prefill +
    bursts of arriving 500-token short chat requests throughout the 1M prefill
    (demonstrating bounded ~9.7% long-prompt overhead with no 10 s cliff).

Standard library only.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STALLS_DIR = ROOT / "experiments" / "2026-10-26-concurrency-stalls"
if str(STALLS_DIR) not in sys.path:
    sys.path.insert(0, str(STALLS_DIR))

import stallsim  # noqa: E402
from stallsim import Engine, Lineage, Policy, Shape, percentile  # noqa: E402


def load_shapes() -> dict[str, Shape]:
    raw = json.loads((STALLS_DIR / "calibration.json").read_text())
    return {name: Shape(**cfg) for name, cfg in raw.items()}


def policy_definitions() -> dict[str, Policy]:
    return {
        "production": Policy(name="production"),
        "prefill-yield-short": Policy(
            name="prefill-yield-short",
            order="bypass",
            first_token_priority=True,
            long_wait_cap_s=10.0,
        ),
        "interleave-k1": Policy(
            name="interleave-k1",
            interleave_decode_steps=1,
        ),
        "interleave-k2": Policy(
            name="interleave-k2",
            interleave_decode_steps=2,
        ),
    }


def _collect_during_prefill_metrics(
    engine: Engine,
    doc: stallsim.Request,
    baseline_chunks: int | None = None,
) -> dict[str, Any]:
    """Compute stream ITL, decode throughput, and chunk counts during doc prefill."""
    doc_start = doc.admitted if doc.admitted is not None else doc.sent
    doc_end = doc.prefilled if doc.prefilled is not None else (doc.first or engine.now)
    prefill_window_s = max(doc_end - doc_start, 1e-6)

    during_gaps: list[float] = []
    for req in engine.requests:
        if req.cls != "stream" or req.first is None:
            continue
        t = req.first
        for gap in req.gaps:
            t_next = t + gap
            if t_next > doc_start + 1e-6 and t < doc_end - 1e-6:
                during_gaps.append(gap)
            t = t_next

    steps_during = [
        s for s in engine.steps
        if s.start >= doc_start - 1e-6 and s.start < doc_end - 1e-6
    ]
    decode_tokens_during = sum(s.emitted for s in steps_during)
    graphed_decode_steps = sum(1 for s in steps_during if s.prefill_tokens == 0)
    doc_chunk_steps = sum(1 for s in steps_during if s.prefill_tokens >= 900)
    extra_moe_passes = (
        max(0, doc_chunk_steps - baseline_chunks)
        if baseline_chunks is not None
        else 0
    )

    probes = [r for r in engine.requests if r.cls == "probe" and r.first is not None]
    probe_ttfts = [r.first - r.sent for r in probes]

    mean_itl = sum(during_gaps) / len(during_gaps) if during_gaps else 0.0
    return {
        "doc_ttft_s": round((doc.first - doc.sent) if doc.first is not None else 0.0, 4),
        "doc_prefill_window_s": round(prefill_window_s, 4),
        "doc_chunk_steps": doc_chunk_steps,
        "graphed_decode_steps_during_prefill": graphed_decode_steps,
        "extra_moe_passes": extra_moe_passes,
        "decode_tok_s_during_prefill": round(decode_tokens_during / prefill_window_s, 2),
        "stream_itl_mean_s": round(mean_itl, 4),
        "stream_itl_p50_s": round(percentile(during_gaps, 0.5) or 0.0, 4),
        "stream_itl_p90_s": round(percentile(during_gaps, 0.9) or 0.0, 4),
        "stream_itl_p99_s": round(percentile(during_gaps, 0.99) or 0.0, 4),
        "stream_itl_max_s": round(max(during_gaps) if during_gaps else 0.0, 4),
        "probe_count": len(probes),
        "probe_ttft_mean_s": (
            round(sum(probe_ttfts) / len(probe_ttfts), 4) if probe_ttfts else None
        ),
        "probe_ttft_p50_s": (
            round(percentile(probe_ttfts, 0.5), 4) if probe_ttfts else None
        ),
        "probe_ttft_p99_s": (
            round(percentile(probe_ttfts, 0.99), 4) if probe_ttfts else None
        ),
    }


def run_stall_scenario(
    shape: Shape,
    policy: Policy,
    *,
    num_streams: int,
    doc_tokens: int,
    probe_offsets: tuple[float, ...] = (),
    probe_prompt: int = 500,
    probe_out: int = 16,
    seed: int = 0,
    baseline_chunks: int | None = None,
) -> dict[str, Any]:
    engine = Engine(shape, policy, seed=seed)
    for i in range(num_streams):
        engine.submit("stream", i * 0.02, 64, 32_000, Lineage())
    doc_sent = 5.0
    doc = engine.submit("doc", doc_sent, doc_tokens, 1, Lineage())
    for dt in probe_offsets:
        engine.submit("probe", doc_sent + dt, probe_prompt, probe_out, Lineage())
    until = doc_sent + doc_tokens / 1200.0 + 120.0
    engine.run(until)
    return _collect_during_prefill_metrics(engine, doc, baseline_chunks=baseline_chunks)


def evaluate_all() -> dict[str, Any]:
    shapes = load_shapes()
    policies = policy_definitions()

    scenarios = {
        "stall-15": {
            "shape": "tp4-ring",
            "description": (
                "15 decoding streams + 131K prefill on tp4-ring (16/16 sequence slots "
                "occupied; admission_open is False so --prefill-yield short cannot trigger)"
            ),
            "num_streams": 15,
            "doc_tokens": 131_072,
            "probe_offsets": (),
        },
        "stall-11": {
            "shape": "tp4-ring",
            "description": (
                "11 decoding streams + 131K prefill + 5 arriving 500-token short probes "
                "on tp4-ring"
            ),
            "num_streams": 11,
            "doc_tokens": 131_072,
            "probe_offsets": (2.0, 5.0, 8.0, 11.0, 14.0),
        },
        "stall-7-tp3": {
            "shape": "tp3-triangle",
            "description": (
                "7 decoding streams + 131K prefill on tp3-triangle (8/8 sequence slots "
                "occupied; admission_open is False)"
            ),
            "num_streams": 7,
            "doc_tokens": 131_072,
            "probe_offsets": (),
        },
        "stall-3-tp3": {
            "shape": "tp3-triangle",
            "description": (
                "3 decoding streams + 131K prefill + 5 arriving 500-token short probes "
                "on tp3-triangle"
            ),
            "num_streams": 3,
            "doc_tokens": 131_072,
            "probe_offsets": (2.0, 5.0, 8.0, 11.0, 14.0),
        },
        "stall-1m": {
            "shape": "tp4-ring",
            "description": (
                "11 decoding streams + 1,000,000-token prefill + 8 arriving 500-token "
                "short chat probes across the 1M prefill on tp4-ring"
            ),
            "num_streams": 11,
            "doc_tokens": 1_000_000,
            "probe_offsets": (10.0, 30.0, 55.0, 80.0, 105.0, 130.0, 155.0, 180.0),
        },
    }

    report: dict[str, Any] = {
        "experiment": "2026-10-30-prefill-decode-interleave",
        "AnalyticalBound": {},
        "scenarios": {},
    }

    tp4 = shapes["tp4-ring"]
    t_dec_15 = tp4.t_dec(15)
    t_chunk_15 = tp4.step_cost(15, [(tp4.threshold, 32_768)])
    for k in (1, 2):
        overhead_ratio = (k * t_dec_15) / t_chunk_15
        report["AnalyticalBound"][f"k={k}"] = {
            "k": k,
            "t_decode_15_ms": round(t_dec_15 * 1000.0, 2),
            "t_chunk_15_ms": round(t_chunk_15 * 1000.0, 2),
            "bounded_long_prompt_overhead_pct": round(overhead_ratio * 100.0, 2),
            "extra_moe_weight_passes": 0,
            "decode_steps_per_cycle": 1 + k,
        }

    for sc_name, cfg in scenarios.items():
        shape = shapes[cfg["shape"]]
        sc_results: dict[str, Any] = {}
        base_metrics = run_stall_scenario(
            shape,
            policies["production"],
            num_streams=cfg["num_streams"],
            doc_tokens=cfg["doc_tokens"],
            probe_offsets=cfg["probe_offsets"],
        )
        base_chunks = base_metrics["doc_chunk_steps"]
        base_doc_ttft = base_metrics["doc_ttft_s"]
        base_dec_tps = base_metrics["decode_tok_s_during_prefill"]
        base_itl_mean = base_metrics["stream_itl_mean_s"]

        for pol_name, policy in policies.items():
            metrics = (
                base_metrics
                if pol_name == "production"
                else run_stall_scenario(
                    shape,
                    policy,
                    num_streams=cfg["num_streams"],
                    doc_tokens=cfg["doc_tokens"],
                    probe_offsets=cfg["probe_offsets"],
                    baseline_chunks=base_chunks,
                )
            )
            metrics["doc_ttft_overhead_pct"] = round(
                ((metrics["doc_ttft_s"] / base_doc_ttft) - 1.0) * 100.0, 2
            )
            metrics["decode_throughput_speedup_x"] = round(
                metrics["decode_tok_s_during_prefill"] / max(base_dec_tps, 1e-6), 2
            )
            metrics["stream_itl_mean_reduction_pct"] = round(
                (1.0 - (metrics["stream_itl_mean_s"] / max(base_itl_mean, 1e-6))) * 100.0,
                2,
            )
            sc_results[pol_name] = metrics

        report["scenarios"][sc_name] = {
            "shape": cfg["shape"],
            "description": cfg["description"],
            "policies": sc_results,
        }

    return report


def main() -> int:
    report = evaluate_all()
    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    for sc_name, sc_data in report["scenarios"].items():
        print(f"\n=== {sc_name} ({sc_data['shape']}) ===")
        for pol_name, m in sc_data["policies"].items():
            probe_str = (
                f" | probe TTFT mean={m['probe_ttft_mean_s']:.2f}s"
                if m["probe_ttft_mean_s"] is not None
                else ""
            )
            print(
                f"  {pol_name:<20}: decode={m['decode_tok_s_during_prefill']:6.1f} tok/s "
                f"({m['decode_throughput_speedup_x']:.2f}x) | "
                f"ITL mean={m['stream_itl_mean_s']:.3f}s ({m['stream_itl_mean_reduction_pct']:+5.1f}%), "
                f"p50={m['stream_itl_p50_s']:.3f}s | "
                f"doc TTFT={m['doc_ttft_s']:.2f}s ({m['doc_ttft_overhead_pct']:+5.1f}%), "
                f"extra_F={m['extra_moe_passes']}{probe_str}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
