"""Replay request mixes under production's scheduling and under each candidate fix.

Every policy is exact by construction or by a named gate (README, "Fixes"). Mixes:
- poisson: the release suite's Poisson step (0.7 req/s ring, 0.42 triangle; 20-token prompts,
  256 tokens out);
- stall / stall-1m: servebench stall with a cold 131K (1M) document and short probes
  (500-token prompts, 128 tokens out) sent during its prefill, beside 11 decoding streams on
  the ring and 4 on the triangle, so the probes find free slots;
- chat-docs: interactive chat (Poisson) beside cold documents of 32K-524K (ring) or 32K-262K;
- agents-cN: AgentX-like sessions (workloads.AgentParams), N lanes, closed loop;
- agents+chat: agent lanes beside interactive chat;
- soak: the 2026-10-23 soak (agents resending 16-48K tool results to 960K) beside chat streams.

Writes runs/results.json (every mix x policy, seeds pooled) and runs/results.txt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibrate  # noqa: E402
import rank  # noqa: E402
import stallsim  # noqa: E402
import workloads  # noqa: E402
from stallsim import Engine, Lineage, Policy, percentile  # noqa: E402

PROBE_PROMPT = 500


# ---------------------------------------------------------------------------------------------
# Mixes: each builds the load on an engine and returns (window start, window end, run until)


def mix_poisson(engine: Engine, rng: random.Random, shape: str):
    rate = 0.7 if shape == "tp4-ring" else 0.42
    workloads.chat_poisson(engine, rng, rate, 0.0, 660.0)
    return 60.0, 660.0, 1200.0


def stall_mix(size: int, probes: tuple):
    def build(engine: Engine, rng: random.Random, shape: str):
        # Slots for the document and the probes in flight: 11 + 1 + 4 of 16, 4 + 1 + 3 of 8.
        streams = 11 if shape == "tp4-ring" else 4
        workloads.stall_probe(engine, streams, size, probes=probes, probe_prompt=PROBE_PROMPT, probe_out=16)
        return 0.0, 20.0 + max(probes) + 1.0, 20.0 + size / 1500 + 120
    return build


def mix_chat_docs(engine: Engine, rng: random.Random, shape: str):
    ring = shape == "tp4-ring"
    workloads.chat_poisson(engine, rng, 0.25 if ring else 0.12, 0.0, 2100.0,
                           prompt=workloads.chat_mix_prompt, out=workloads.chat_mix_out)
    workloads.documents(engine, rng, 1 / 150 if ring else 1 / 240, 0.0, 2100.0,
                        high=524288 if ring else 262144)
    return 300.0, 2100.0, 3600.0


def agents_mix(lanes: int, chat_rate: float = 0.0):
    def build(engine: Engine, rng: random.Random, shape: str):
        workloads.agentx(engine, rng, lanes, 0.0, 4500.0)
        if chat_rate:
            workloads.chat_poisson(engine, rng, chat_rate, 0.0, 4500.0,
                                   prompt=workloads.chat_mix_prompt, out=workloads.chat_mix_out)
        return 900.0, 4500.0, 6000.0
    return build


def mix_soak(engine: Engine, rng: random.Random, shape: str):
    ring = shape == "tp4-ring"
    workloads.soak_agents(engine, rng, 4 if ring else 2, 0.0, 2100.0)
    workloads.closed_loop(engine, rng, 8 if ring else 4, 0.0, 2100.0, stagger=0.5)
    return 300.0, 2100.0, 3000.0


MIXES = {
    "tp4-ring": {
        "poisson": (mix_poisson, 8),
        "stall": (stall_mix(131072, (2.0, 5.0, 8.0, 11.0, 14.0)), 6),
        "stall-1m": (stall_mix(1000000, (10.0, 40.0, 80.0, 120.0, 160.0)), 3),
        "chat-docs": (mix_chat_docs, 8),
        "agents-c8": (agents_mix(8), 6),
        "agents-c16": (agents_mix(16), 6),
        "agents-c24": (agents_mix(24), 4),
        "agents-c8+chat": (agents_mix(8, 0.12), 8),
        "agents-c8+chat-heavy": (agents_mix(8, 0.3), 4),
        "soak": (mix_soak, 8),
    },
    "tp3-triangle": {
        "poisson": (mix_poisson, 8),
        "stall": (stall_mix(131072, (2.0, 5.0, 8.0, 11.0, 14.0)), 6),
        "chat-docs": (mix_chat_docs, 8),
        "agents-c4": (agents_mix(4), 8),
        "agents-c2+chat": (agents_mix(2, 0.06), 8),
        "soak": (mix_soak, 8),
    },
}


# ---------------------------------------------------------------------------------------------
# Policies (README, "Fixes")


def policies(shape: str) -> dict:
    ring = shape == "tp4-ring"
    grid = 2048
    top = 16384 if ring else 8192

    def tiers(*levels):
        return [(n, t) for n, t in levels]

    p = {
        "production": Policy(),
        "sink-off*": Policy(sink_hold=False),
        "engram-prefetch": Policy(engram_prefetch=True),
        "overhead-1.5ms*": Policy(overhead_cut_s=0.0015),
        "decode-rows-graphed*": Policy(alpha_scale=1.0 / max(calibrate.load_shapes()[shape].alpha, 1.0)),
        "bypass (no aging)": Policy(order="bypass"),
        "bypass": Policy(order="bypass", long_wait_cap_s=10.0),
        "bypass-ft": Policy(order="bypass", first_token_priority=True, long_wait_cap_s=10.0),
        "spf": Policy(order="spf", long_wait_cap_s=10.0),
        "lanes2 (vLLM)": Policy(order="rr2"),
        "lanes2-release": Policy(order="rr2", lane_release=True),
        "tiers-8k": Policy(chunk_tiers=tiers((0, top), (1, 8192)), grid=grid) if ring else None,
        "tiers-4k": Policy(chunk_tiers=tiers((0, top), (1, 4096)), grid=grid),
        "tiers-2k": Policy(chunk_tiers=tiers((0, top), (1, 2048)), grid=grid),
        "tiers-by-load": Policy(chunk_tiers=tiers((0, top), (1, 8192 if ring else 4096), (5, 4096 if ring else 2048),
                                                   (11 if ring else 6, 2048)), grid=grid),
        "alone-32k+4k": Policy(chunk_tiers=tiers((0, 32768), (1, 4096)), grid=grid, budget_alone=32768) if ring else None,
        "alone-12k+4k": None if ring else Policy(chunk_tiers=tiers((0, 12288), (1, 4096)), grid=grid, budget_alone=12288),
        "bypass-ft+tiers-4k": Policy(order="bypass", first_token_priority=True, long_wait_cap_s=10.0,
                                     chunk_tiers=tiers((0, top), (1, 4096)), grid=grid),
        "bypass-ft+slots": Policy(order="bypass", first_token_priority=True, long_wait_cap_s=10.0,
                                  max_seqs=24 if ring else 12),
        "bypass-ft+slots+prefix": Policy(order="bypass", first_token_priority=True, long_wait_cap_s=10.0,
                                         max_seqs=24 if ring else 12, eviction="prefix",
                                         retention_interval=None if ring else 2048),
        "prefix-eviction": Policy(eviction="prefix"),
        "pool-2x": Policy(pool_scale=2.0) if ring else None,
        "pool+1.3GiB": None if ring else Policy(extra_pool_blocks=6126),
        "retention-8192": None if ring else Policy(retention_interval=8192),
        "retention-2048": None if ring else Policy(retention_interval=2048),
        "slots-24": Policy(max_seqs=24) if ring else None,
        "slots-12": None if ring else Policy(max_seqs=12),
        "slots-24 (steeper)*": Policy(max_seqs=24, dec_extrapolation=2.0) if ring else None,
        "slots-12 (steeper)*": None if ring else Policy(max_seqs=12, dec_extrapolation=2.0),
        "slots-32": Policy(max_seqs=32) if ring else None,
    }
    out = {}
    for name, policy in p.items():
        if policy is not None:
            out[name] = dataclasses.replace(policy, name=name)
    return out


# ---------------------------------------------------------------------------------------------
# One run


def alone_ttft(shape: stallsim.Shape, request) -> float:
    """The request's TTFT on an idle engine with the same hit: frontend, chunks, held step."""
    threshold = shape.threshold
    x, total, first = request.hit_tokens, 0.0, True
    while x < request.prompt:
        chunk = min(threshold - x % threshold, request.prompt - x)
        total += shape.step_cost(0, [(chunk, x)], engram_s=shape.engram_first_s if first and chunk >= 1024 else 0.0)
        x += chunk
        first = False
    held = shape.t_dec(1) if request.max_out > 1 else 0.0
    return shape.fe0 + shape.fe1 * request.prompt + shape.host_s + total + held


def run_one(args: tuple) -> dict:
    shape_name, mix_name, policy_name, seed = args
    shape = calibrate.load_shapes()[shape_name]
    policy = policies(shape_name)[policy_name]
    build, _ = MIXES[shape_name][mix_name]
    engine = Engine(shape, policy, seed=seed)
    start, end, until = build(engine, random.Random(1000 + seed), shape_name)
    started = time.time()
    engine.run(until)
    summary = stallsim.summarize(engine, start, end)
    classes = {}
    for cls in summary["classes"]:
        requests = [r for r in engine.requests if r.cls == cls and start <= r.sent < end]
        done = [r for r in requests if r.first is not None]
        classes[cls] = {
            "ttft": [round(r.first - r.sent, 4) for r in done],
            "stretch": [round((r.first - r.sent) / alone_ttft(shape, r), 3) for r in done],
            "slo": [r.first - r.sent <= 2.0 and (r.emitted < 2 or r.finished_at is None
                    or (r.last - r.first) / (r.emitted - 1) <= 0.1) for r in done],
            "tpot": [round((r.last - r.first) / (r.emitted - 1), 5) for r in done if r.finished_at and r.emitted > 1],
            "itl_p50": summary["classes"][cls]["itl_p50"], "itl_p99": summary["classes"][cls]["itl_p99"],
            "itl_max": summary["classes"][cls]["itl_max"],
            "unfinished": summary["classes"][cls]["unfinished"], "requests": len(requests),
            "hit_share": summary["classes"][cls]["hit_share"],
            "ideal_hit_share": summary["classes"][cls]["ideal_hit_share"],
        }
    # decode gaps of the streams while the stall document prefills
    during = []
    doc = next((r for r in engine.requests if r.cls == "doc" and r.max_out == 1), None)
    if doc is not None and doc.first is not None:
        for r in engine.requests:
            if r.cls != "stream" or r.first is None:
                continue
            t = r.first
            for gap in r.gaps:
                t += gap
                if t > doc.sent and t - gap < doc.first:
                    during.append(gap)
    return {
        "shape": shape_name, "mix": mix_name, "policy": policy_name, "seed": seed,
        "seconds": round(time.time() - started, 2),
        "output_tok_s": summary["output_tok_s"], "prefill_tok_s": summary["prefill_tok_s"],
        "recompute_tok_s": summary["recompute_tok_s"], "preemptions": summary["preemptions"],
        "evicted_blocks": summary["evicted_blocks"], "evicted_checkpoints": summary["evicted_checkpoints"],
        "long_step_time_share": summary["long_step_time_share"],
        "livelocked": summary["livelocked"],
        "doc_ttft": (doc.first - doc.sent) if doc is not None and doc.first else None,
        "stall_p50": percentile(during, 0.5), "stall_p99": percentile(during, 0.99),
        "stall_worst": max(during) if during else None,
        "classes": classes,
    }


# ---------------------------------------------------------------------------------------------
# Pooling and tables


def pool(runs: list) -> dict:
    """Seeds pooled: throughput means, latency percentiles over every seed's requests."""
    out = {"seeds": len(runs)}
    out["livelocked"] = sum(1 for r in runs if r.get("livelocked"))
    for key in ("output_tok_s", "prefill_tok_s", "recompute_tok_s", "preemptions", "evicted_blocks",
                "evicted_checkpoints", "long_step_time_share", "doc_ttft", "stall_p50", "stall_p99", "stall_worst"):
        values = [r[key] for r in runs if r[key] is not None]
        out[key] = sum(values) / len(values) if values else None
    classes = {}
    for cls in sorted({c for r in runs for c in r["classes"]}):
        rows = [r["classes"][cls] for r in runs if cls in r["classes"]]
        ttft = [v for row in rows for v in row["ttft"]]
        stretch = [v for row in rows for v in row["stretch"]]
        tpot = [v for row in rows for v in row["tpot"]]

        def mean_of(key):
            values = [row[key] for row in rows if row[key] is not None]
            return sum(values) / len(values) if values else None

        slo = [v for row in rows for v in row["slo"]]
        classes[cls] = {
            "requests": sum(row["requests"] for row in rows), "unfinished": sum(row["unfinished"] for row in rows),
            "ttft_p50": percentile(ttft, 0.5), "ttft_p90": percentile(ttft, 0.9), "ttft_p99": percentile(ttft, 0.99),
            "ttft_mean": sum(ttft) / len(ttft) if ttft else None,
            "stretch_p50": percentile(stretch, 0.5), "stretch_p90": percentile(stretch, 0.9),
            "stretch_p99": percentile(stretch, 0.99),
            "tpot_p50": percentile(tpot, 0.5), "tpot_p99": percentile(tpot, 0.99),
            "itl_p50": mean_of("itl_p50"), "itl_p99": mean_of("itl_p99"), "itl_max": mean_of("itl_max"),
            "slo_share": sum(slo) / len(slo) if slo else None,
            "hit_share": mean_of("hit_share"), "ideal_hit_share": mean_of("ideal_hit_share"),
        }
    out["classes"] = classes
    return out


def paired_intervals(base: list, runs: list, user: str, draws: int = 400, seed: int = 0) -> dict:
    """95% intervals of the ratios fix / production, resampling seeds with replacement and
    taking the same seeds from both (the workload is generated from the seed)."""
    import random as random_
    rng = random_.Random(seed)
    by_seed = {r["seed"]: r for r in base}
    pairs = [(by_seed[r["seed"]], r) for r in runs if r["seed"] in by_seed]
    if not pairs:
        return {}
    samples = {"output": [], "ttft_p50": [], "ttft_p99": []}
    for _ in range(draws):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        b_out = sum(p[0]["output_tok_s"] for p in chosen)
        f_out = sum(p[1]["output_tok_s"] for p in chosen)
        if b_out > 0:
            samples["output"].append(f_out / b_out)
        b_t = [v for p in chosen for v in p[0]["classes"].get(user, {}).get("ttft", [])]
        f_t = [v for p in chosen for v in p[1]["classes"].get(user, {}).get("ttft", [])]
        for q, key in ((0.5, "ttft_p50"), (0.99, "ttft_p99")):
            bq, fq = percentile(b_t, q), percentile(f_t, q)
            if bq and fq:
                samples[key].append(fq / bq)
    return {key: [percentile(v, 0.025), percentile(v, 0.975)] for key, v in samples.items() if v}


def fmt(value, scale=1.0, digits=2, unit=""):
    if value is None:
        return "-"
    return f"{value * scale:.{digits}f}{unit}"


def table(results: dict) -> str:
    lines = []
    for shape, mixes in results.items():
        for mix, by_policy in mixes.items():
            base = by_policy["production"]
            lines.append(f"\n=== {shape} / {mix} ({base['seeds']} seeds pooled)")
            classes = [c for c in ("chat", "probe", "stream", "agent", "agent-sub", "agent-side", "doc", "soak")
                       if c in base["classes"]]
            header = f"{'policy':<22} {'out tok/s':>9} {'recomp':>7} {'preempt':>7} {'change':>7}"
            for c in classes:
                header += f" | {c[:9]:>9} TTFT p50/p99 s, ITL p99 s"
            if base.get("stall_p50") is not None:
                header += " | stall p50/worst s, doc TTFT s"
            lines.append(header)
            for policy, row in by_policy.items():
                change = (row["output_tok_s"] / base["output_tok_s"] - 1) * 100 if base["output_tok_s"] else 0.0
                line = f"{policy:<22} {row['output_tok_s']:9.1f} {fmt(row['recompute_tok_s'], digits=0):>7} {fmt(row['preemptions'], digits=0):>7}"
                line += f" {change:+6.1f}%" if policy != "production" else "        "
                if row.get("livelocked"):
                    line += f" LIVELOCK x{row['livelocked']}"
                for c in classes:
                    k = row["classes"].get(c, {})
                    line += (f" | {fmt(k.get('ttft_p50')):>6}/{fmt(k.get('ttft_p99')):>6}"
                             f" {fmt(k.get('itl_p99'), digits=3):>6}")
                if base.get("stall_p50") is not None:
                    line += f" | {fmt(row['stall_p50'])}/{fmt(row['stall_worst'])} {fmt(row['doc_ttft'], digits=1)}"
                lines.append(line)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shape", action="append", help="default: both")
    parser.add_argument("--mix", action="append", help="default: every mix")
    parser.add_argument("--policy", action="append", help="default: every policy")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", default=str(HERE / "runs/results.json"))
    parser.add_argument("--table", default=str(HERE / "runs/results.txt"))
    args = parser.parse_args(argv)
    jobs = []
    for shape in args.shape or list(MIXES):
        for mix, (_, seeds) in MIXES[shape].items():
            if args.mix and mix not in args.mix:
                continue
            for policy in policies(shape):
                if args.policy and policy not in args.policy and policy != "production":
                    continue
                jobs.extend((shape, mix, policy, seed) for seed in range(seeds))
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    runs = []
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool_:
        for index, result in enumerate(pool_.map(run_one, jobs, chunksize=1)):
            runs.append(result)
            if (index + 1) % 50 == 0:
                print(f"  {index + 1}/{len(jobs)} ({time.time() - started:.0f} s)", flush=True)
    results: dict = {}
    for shape in dict.fromkeys(r["shape"] for r in runs):
        results[shape] = {}
        for mix in dict.fromkeys(r["mix"] for r in runs if r["shape"] == shape):
            results[shape][mix] = {}
            for policy in dict.fromkeys(r["policy"] for r in runs if r["shape"] == shape and r["mix"] == mix):
                chosen = [r for r in runs if (r["shape"], r["mix"], r["policy"]) == (shape, mix, policy)]
                results[shape][mix][policy] = pool(chosen)
                if policy != "production":
                    base = [r for r in runs if (r["shape"], r["mix"], r["policy"]) == (shape, mix, "production")]
                    user = rank.KEYS[mix][0]
                    results[shape][mix][policy]["intervals"] = paired_intervals(base, chosen, user)
    Path(args.out).write_text(json.dumps(results, separators=(",", ":")) + "\n")
    text = table(results)
    Path(args.table).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
