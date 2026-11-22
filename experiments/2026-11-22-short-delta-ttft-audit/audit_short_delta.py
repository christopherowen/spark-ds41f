#!/usr/bin/env python3
"""Measure exact live TTFT breakdown (host, queue, GPU prefill) for cached short deltas.

Pure Python stdlib (zero third-party deps). Uses vLLM's --enable-per-request-metrics
and streaming SSE on /v1/chat/completions to separate:
  - client_ttft_ms: request send -> first content/reasoning token chunk
  - server_queue_ms: EngineCore queued_ts -> scheduled_ts
  - server_ttft_ms:  EngineCore scheduled_ts -> first_token_ts (GPU prefill + sink hold)
  - host_overhead_ms: client_ttft_ms - (server_queue_ms + server_ttft_ms)
  - cached_tokens / prompt_tokens (verifying prefix-cache hits)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import random
import statistics
import time
import urllib.parse
from dataclasses import asdict, dataclass


@dataclass
class ReqResult:
    scenario: str
    idx: int
    stagger_ms: float
    prompt_tokens: int
    cached_tokens: int
    uncached_tokens: int
    client_ttft_ms: float
    client_e2e_ms: float
    server_queue_ms: float | None
    server_ttft_ms: float | None
    host_overhead_ms: float | None
    first_token_text: str
    error: str | None = None


def make_text(seed: int, words: int) -> str:
    rng = random.Random(seed)
    vocab = [
        "kernel", "matrix", "dispatch", "scheduler", "prefill", "decode", "block",
        "cache", "offset", "stride", "buffer", "latency", "window", "stream",
        "tensor", "expert", "router", "weight", "metric", "budget", "chunk",
        "prefix", "memory", "thread", "shared", "verify", "draft", "token",
        "layer", "vector", "norm", "scale", "index", "table", "queue", "batch",
    ]
    out = []
    for i in range(words):
        w = vocab[rng.randrange(len(vocab))]
        if i % 7 == 0:
            out.append(f"{w}_{rng.randrange(1000, 9999)}")
        else:
            out.append(w)
    return " ".join(out)


def send_chat(
    host: str,
    port: int,
    model: str,
    messages: list[dict],
    max_tokens: int,
    scenario: str,
    idx: int,
    stagger_ms: float = 0.0,
) -> ReqResult:
    if stagger_ms > 0:
        time.sleep(stagger_ms / 1000.0)
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }).encode("utf-8")
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    first_tok = ""
    prompt_tokens = 0
    cached_tokens = 0
    server_queue_ms: float | None = None
    server_ttft_ms: float | None = None
    conn = http.client.HTTPConnection(host, port, timeout=600)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            txt = resp.read().decode("utf-8", errors="replace")
            return ReqResult(
                scenario, idx, stagger_ms, 0, 0, 0, 0, 0, None, None, None, "",
                f"HTTP {resp.status}: {txt[:200]}",
            )
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if ttft_ms is None:
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or ""
                    )
                    if piece:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                        first_tok = piece[:24]
            usage = obj.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                ptd = usage.get("prompt_tokens_details") or {}
                if isinstance(ptd, dict) and ptd.get("cached_tokens") is not None:
                    cached_tokens = int(ptd["cached_tokens"])
            metrics = obj.get("metrics")
            if isinstance(metrics, dict):
                if metrics.get("queue_time_ms") is not None:
                    server_queue_ms = float(metrics["queue_time_ms"])
                if metrics.get("time_to_first_token_ms") is not None:
                    server_ttft_ms = float(metrics["time_to_first_token_ms"])
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        ttft = ttft_ms if ttft_ms is not None else e2e_ms
        host_oh = (
            ttft - (server_queue_ms + server_ttft_ms)
            if server_queue_ms is not None and server_ttft_ms is not None
            else None
        )
        return ReqResult(
            scenario=scenario,
            idx=idx,
            stagger_ms=stagger_ms,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            uncached_tokens=max(0, prompt_tokens - cached_tokens),
            client_ttft_ms=round(ttft, 2),
            client_e2e_ms=round(e2e_ms, 2),
            server_queue_ms=round(server_queue_ms, 2) if server_queue_ms is not None else None,
            server_ttft_ms=round(server_ttft_ms, 2) if server_ttft_ms is not None else None,
            host_overhead_ms=round(host_oh, 2) if host_oh is not None else None,
            first_token_text=first_tok,
        )
    except Exception as exc:
        return ReqResult(scenario, idx, stagger_ms, 0, 0, 0, 0, 0, None, None, None, "", str(exc))
    finally:
        conn.close()


def summarize(name: str, rows: list[ReqResult]) -> dict:
    ok = [r for r in rows if not r.error]
    errs = [r.error for r in rows if r.error]

    def med(vals):
        v = [x for x in vals if x is not None]
        return round(statistics.median(v), 1) if v else None

    def mx(vals):
        v = [x for x in vals if x is not None]
        return round(max(v), 1) if v else None

    s = {
        "scenario": name,
        "n": len(ok),
        "errors": len(errs),
        "prompt_tokens_med": med([r.prompt_tokens for r in ok]),
        "cached_tokens_med": med([r.cached_tokens for r in ok]),
        "uncached_tokens_med": med([r.uncached_tokens for r in ok]),
        "client_ttft_p50_ms": med([r.client_ttft_ms for r in ok]),
        "client_ttft_max_ms": mx([r.client_ttft_ms for r in ok]),
        "server_queue_p50_ms": med([r.server_queue_ms for r in ok]),
        "server_queue_max_ms": mx([r.server_queue_ms for r in ok]),
        "server_ttft_p50_ms": med([r.server_ttft_ms for r in ok]),
        "server_ttft_max_ms": mx([r.server_ttft_ms for r in ok]),
        "host_overhead_p50_ms": med([r.host_overhead_ms for r in ok]),
    }
    print(
        f"{name:<28} n={s['n']:<2} err={s['errors']} prompt={str(s['prompt_tokens_med']):>7} "
        f"cached={str(s['cached_tokens_med']):>7} uncached={str(s['uncached_tokens_med']):>6} | "
        f"client={str(s['client_ttft_p50_ms']):>6}ms (max {str(s['client_ttft_max_ms']):>6}) | "
        f"queue={str(s['server_queue_p50_ms']):>6}ms (max {str(s['server_queue_max_ms']):>6}) | "
        f"gpu={str(s['server_ttft_p50_ms']):>6}ms | host={str(s['host_overhead_p50_ms']):>5}ms"
    )
    if errs:
        print(f"  ERRORS: {errs[:2]}")
    return s


def run_audit(url: str, model: str, out_path: str, chunk_threshold: int) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "10.0.1.71"
    port = parsed.port or 8000
    all_results: list[ReqResult] = []
    summaries: list[dict] = []

    # Shared ~62K-token base prefix (safely below 11 * 6048 = 66528)
    base_prefix = make_text(seed=1001, words=39500)
    warm = send_chat(
        host, port, model,
        [{"role": "system", "content": "Reply with one word: OK."}, {"role": "user", "content": base_prefix}],
        max_tokens=4, scenario="warmup-62k", idx=0,
    )
    print(
        f"Warmup 62K: prompt={warm.prompt_tokens} (mod {chunk_threshold} = {warm.prompt_tokens % chunk_threshold}) "
        f"cached={warm.cached_tokens} client={warm.client_ttft_ms}ms gpu={warm.server_ttft_ms}ms "
        f"queue={warm.server_queue_ms}ms host={warm.host_overhead_ms}ms err={warm.error}"
    )
    all_results.append(warm)

    # Scenario A: Solo cached-delta ladder {128, 256, 512, 1024, 2048} uncached tokens
    for delta_words, label in [
        (75, "A-solo-d128"),
        (160, "A-solo-d256"),
        (330, "A-solo-d512"),
        (660, "A-solo-d1024"),
        (1320, "A-solo-d2048"),
    ]:
        rows = []
        for r in range(3):
            delta = f"\nTurn {label}-{r}: " + make_text(seed=2000 + delta_words * 10 + r, words=delta_words)
            rows.append(send_chat(
                host, port, model,
                [{"role": "system", "content": "Reply with one word: OK."}, {"role": "user", "content": base_prefix + delta}],
                max_tokens=8, scenario=label, idx=r,
            ))
        all_results.extend(rows)
        summaries.append(summarize(label, rows))

    # Scenario B: Concurrent short-delta waves (4 and 8 concurrent, 0ms vs 25ms stagger)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for conc, stagger_ms, delta_words, label in [
            (4, 0.0, 220, "B-wave4-d350-sync"),
            (4, 25.0, 220, "B-wave4-d350-stag25ms"),
            (8, 0.0, 220, "B-wave8-d350-sync"),
            (8, 25.0, 220, "B-wave8-d350-stag25ms"),
        ]:
            t_wave0 = time.perf_counter()
            futs = []
            for i in range(conc):
                delta = f"\nWave {label}-{i}: " + make_text(
                    seed=5000 + conc * 100 + int(stagger_ms) * 10 + i, words=delta_words
                )
                futs.append(pool.submit(
                    send_chat, host, port, model,
                    [{"role": "system", "content": "Reply with one word: OK."}, {"role": "user", "content": base_prefix + delta}],
                    8, label, i, i * stagger_ms,
                ))
            rows = [f.result() for f in futs]
            wave_ms = (time.perf_counter() - t_wave0) * 1000.0
            all_results.extend(rows)
            s = summarize(label, rows)
            s["wave_wall_ms"] = round(wave_ms, 1)
            print(f"  -> {label} wave_wall={wave_ms:.1f}ms")
            summaries.append(s)

        # Scenario C: Delta straddling next multiple of chunk_threshold vs staying inside
        next_mult = ((warm.prompt_tokens // chunk_threshold) + 1) * chunk_threshold
        target_boundary = next_mult - 400
        extra_words = max(0, int((target_boundary - warm.prompt_tokens) / 1.55))
        boundary_prefix = base_prefix + "\n" + make_text(seed=7777, words=extra_words)
        bw = send_chat(
            host, port, model,
            [{"role": "system", "content": "Reply with one word: OK."}, {"role": "user", "content": boundary_prefix}],
            max_tokens=4, scenario="warmup-boundary", idx=0,
        )
        print(
            f"Warmup boundary: prompt={bw.prompt_tokens} (next mult {next_mult}, "
            f"dist_to_boundary={next_mult - bw.prompt_tokens})"
        )
        rows_b = []
        for r in range(3):
            delta = f"\nBoundary-{r}: " + make_text(seed=8000 + r, words=660)
            rows_b.append(send_chat(
                host, port, model,
                [{"role": "system", "content": "Reply with one word: OK."}, {"role": "user", "content": boundary_prefix + delta}],
                max_tokens=8, scenario="C-straddle-d1024", idx=r,
            ))
        all_results.extend(rows_b)
        summaries.append(summarize("C-straddle-d1024", rows_b))

        # Scenario D: Short delta arriving during 4 active decode streams
        bg_futs = [
            pool.submit(
                send_chat, host, port, model,
                [{"role": "user", "content": f"Write a detailed 500-word essay on topic {i}: " + make_text(seed=9100 + i, words=40)}],
                256, "D-bg-decode", i, 0.0,
            )
            for i in range(4)
        ]
        time.sleep(1.5)
        rows_d = []
        for r in range(4):
            delta = f"\nUnderDecode-{r}: " + make_text(seed=9500 + r, words=330)
            rows_d.append(send_chat(
                host, port, model,
                [{"role": "system", "content": "Reply with one word: OK."}, {"role": "user", "content": base_prefix + delta}],
                max_tokens=8, scenario="D-under-c4-decode-d512", idx=r,
            ))
            time.sleep(0.2)
        for f in bg_futs:
            f.result()
        all_results.extend(rows_d)
        summaries.append(summarize("D-under-c4-decode-d512", rows_d))

    with open(out_path, "w") as f:
        json.dump({"summaries": summaries, "results": [asdict(r) for r in all_results]}, f, indent=2)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.0.1.71:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--out", default="/tmp/short_delta_audit.json")
    ap.add_argument("--chunk-threshold", type=int, default=6048)
    args = ap.parse_args()
    run_audit(args.url, args.model, args.out, args.chunk_threshold)


if __name__ == "__main__":
    main()
