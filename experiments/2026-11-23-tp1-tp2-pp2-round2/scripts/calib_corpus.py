#!/usr/bin/env python3
"""Multi-domain, Sonar-disjoint calibration prompt driver for DS4.1 expert routing counts.

Sends 40 diverse prompts across 5 domains (science/history prose, multi-language
systems code, math/algorithms, structured JSON/tool-calling, and systems/SQL/ops)
to an OpenAI-compatible vLLM endpoint at concurrency 4, generating ~15K decode
tokens (~3.6M layer-expert route observations) with zero overlap with Sonar.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request

PROMPTS: list[tuple[str, str]] = [
    # Domain 1: Science & History Prose
    ("prose", "Explain how plate tectonics drives volcanic island arcs and deep ocean trenches, including the role of slab dehydration in mantle wedge melting."),
    ("prose", "Describe the engineering of Roman concrete (opus caementicium), why pozzolanic ash makes marine structures self-healing, and how it differs from modern Portland cement."),
    ("prose", "Walk through the physiology of human sleep architecture across NREM and REM stages, detailing thalamocortical spindles, slow-wave oscillations, and memory consolidation."),
    ("prose", "Trace the economic mechanisms of the 17th-century Dutch East India Company (VOC), including fractional share issuance, secondary markets, and maritime risk pooling."),
    ("prose", "Explain the physics of atmospheric Hadley, Ferrel, and polar circulation cells, and how the Coriolis effect shapes subtropical jet streams and trade winds."),
    ("prose", "Discuss how cephalopods achieve rapid adaptive camouflage through chromatophores, iridophores, and leucophores, despite possessing monochromatic vision."),
    ("prose", "Summarize the transition from cuneiform accounting tablets in Sumer to alphabetic scripts along the Levantine coast, highlighting cognitive and administrative trade-offs."),
    ("prose", "Explain how C4 and CAM photosynthetic pathways evolved to suppress photorespiration in arid climates, comparing anatomical and temporal carbon fixation."),
    # Domain 2: Multi-Language Systems Code
    ("code", "Write a clean, well-documented Rust implementation of a bounded lock-free single-producer single-consumer (SPSC) ring buffer using Acquire/Release atomics and cache-line padding."),
    ("code", "Implement a C++20 intrusive red-black tree with custom comparator support, const iterators, and exception-safe insertion and erasure."),
    ("code", "Write a Go token-bucket HTTP rate limiter middleware supporting per-client sharded maps, background stale-entry eviction, and graceful context cancellation."),
    ("code", "Build a TypeScript recursive-descent JSON5 parser with line/column diagnostics, trailing commas, hex numbers, and single-quoted strings."),
    ("code", "Write a Python script using only the standard library that parses a SQLite3 database header and WAL file to inspect frame checksums and page numbers."),
    ("code", "Implement a Zig bump/arena allocator with alignment support, scoped checkpoints/rewinds, and unit tests verifying overflow handling."),
    ("code", "Write a CUDA C++ templated block-wide segmented prefix sum (inclusive scan) using warp shuffles (__shfl_up_sync) and minimal shared memory bank conflicts."),
    ("code", "Write a Haskell implementation of an expression evaluator supporting variables, let-bindings, and division-by-zero error propagation via ExceptT and ReaderT."),
    # Domain 3: Math, Proofs & Algorithms
    ("math", "Prove that there are infinitely many primes congruent to 3 modulo 4, and explain why the elementary Euclidean construction fails for primes congruent to 1 modulo 4."),
    ("math", "Derive the Cooley-Tukey radix-2 decimation-in-time Fast Fourier Transform from the DFT definition, analyze its O(N log N) recurrence, and explain bit-reversal permutation."),
    ("math", "Prove the Max-Flow Min-Cut theorem using residual graphs and augmenting paths, and analyze why Dinic's algorithm achieves O(V^2 E) worst-case complexity."),
    ("math", "Derive the closed-form solution to the Gambler's Ruin problem with biased coin probability p != 1/2 using martingale optional stopping, and verify the p -> 1/2 limit."),
    ("math", "Explain the relationship between Singular Value Decomposition (SVD), Eckart-Young-Mirsky low-rank approximation, and Principal Component Analysis (PCA)."),
    ("math", "Formulate the Karush-Kuhn-Tucker (KKT) conditions for convex optimization with inequality constraints, and apply them to derive the water-filling power allocation rule."),
    ("math", "Derive the generating function for Catalan numbers, prove the closed form C_n = (1/(n+1)) binom(2n, n), and give three distinct combinatorial interpretations."),
    ("math", "Analyze the amortized cost of Union-Find with path compression and union by rank, outlining why the inverse Ackermann function alpha(n) appears."),
    # Domain 4: Structured JSON / Tool-Calling / Agent Dialogues
    ("agent", "You are an SRE triage agent. Produce a strict JSON plan (keys: severity, hypotheses, tool_calls, rollback_criteria) to investigate a sudden 5x p99 latency spike in a Redis-backed checkout service."),
    ("agent", "Design an OpenAI-compatible JSON Schema tool catalog for a travel booking assistant with 4 tools: search_flights, hold_itinerary, check_visa_requirements, and issue_ticket."),
    ("agent", "Simulate a 3-turn agent debugging trace in JSON showing tool invocation arguments and observations for diagnosing a Kubernetes CrashLoopBackOff caused by an OOMKilled JVM container."),
    ("agent", "Generate a structured JSON migration plan for zero-downtime splitting of a monolithic Postgres users table into auth_identities and user_profiles, including dual-write and backfill phases."),
    ("agent", "Produce a JSON incident postmortem (timeline, root_cause, contributing_factors, action_items with owners and SLOs) for an expired intermediate TLS certificate on an API gateway."),
    ("agent", "Given a customer order payload with partial refunds and multi-currency tax rules, emit a step-by-step JSON ledger reconciliation output with debit/credit line items."),
    ("agent", "Write a structured JSON compiler diagnostic report for 5 common Rust borrow-checker errors, each with error_code, minimal_repro, explanation, and idiomatic_fix."),
    ("agent", "Create a JSON workflow DAG for an automated ML data-quality gate (schema_check, drift_ks_test, label_leakage_scan, pii_redaction) with retry policies and artifact contracts."),
    # Domain 5: Systems, SQL, Shell & Technical Docs
    ("ops", "Explain PostgreSQL MVCC tuple visibility rules (xmin, xmax, hint bits, visibility map), transaction ID wraparound, and how autovacuum freeze prevents data loss."),
    ("ops", "Compare Linux epoll edge-triggered readiness notification with io_uring submission/completion rings, registered fixed buffers, and SQPOLL for high-IOPS NVMe storage."),
    ("ops", "Walk through the TLS 1.3 full handshake message flow, HKDF key schedule derivation, 0-RTT early-data replay risks, and why RSA key transport was removed."),
    ("ops", "Explain how Raft achieves leader election, log replication safety, and membership changes (joint consensus vs single-server), including why uncommitted terms cannot be committed directly."),
    ("ops", "Detail Git's internal object model (blob, tree, commit, tag), packfile delta chains, multi-pack-index (MIDX), and commit-graph generation numbers."),
    ("ops", "Write an analytical PostgreSQL SQL query using window functions and CTEs to compute 7-day rolling retention cohorts, sessionization with 30-minute inactivity gaps, and top-3 revenue SKUs per region."),
    ("ops", "Explain BGP path selection attributes (LOCAL_PREF, AS_PATH, ORIGIN, MED, eBGP vs iBGP) and how RPKI Route Origin Validation mitigates prefix hijacks."),
    ("ops", "Describe NVMe PCIe architecture: submission and completion queue pairs, MSI-X interrupts, namespace LBA formatting, and O_DIRECT alignment requirements on Linux."),
]


def run_one(base_url: str, model: str, idx: int, domain: str, prompt: str, max_tokens: int) -> tuple[int, str, int, float]:
    t0 = time.monotonic()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    toks = int(data.get("usage", {}).get("completion_tokens", 0))
    return idx, domain, toks, time.monotonic() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    t0 = time.monotonic()
    total_toks = 0
    by_domain: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(run_one, args.base_url, args.model, i, dom, pr, args.max_tokens)
            for i, (dom, pr) in enumerate(PROMPTS)
        ]
        for fut in concurrent.futures.as_completed(futs):
            idx, dom, toks, dt = fut.result()
            total_toks += toks
            by_domain[dom] = by_domain.get(dom, 0) + toks
            print(f"[{idx+1:02d}/{len(PROMPTS)}] {dom:6s}: {toks:4d} tok in {dt:5.1f}s ({toks/max(dt,1e-6):5.1f} tok/s)", flush=True)
    elapsed = time.monotonic() - t0
    print(f"DONE: {total_toks} decode tokens across {len(PROMPTS)} prompts in {elapsed:.1f}s ({total_toks/elapsed:.1f} tok/s)")
    print("By domain:", json.dumps(by_domain, sort_keys=True))


if __name__ == "__main__":
    main()
