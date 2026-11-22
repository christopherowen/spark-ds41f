#!/usr/bin/env python3
"""Realistic multi-turn (20-turn) cached short-delta TTFT & co-admission audit.

Separates:
  - True multi-turn cached deltas (Turns 0..19 unchanged with newlines -> hits
    deepseek_v41 segment_cache in ~2 ms, exposing true EngineCore/GPU co-admission)
  - 4/8-way simultaneous vs 25ms-staggered waves
  - Short delta during c4 active decode AND during a 131K cold prefill
  - 4-way simultaneous cold 32K prompts (measuring renderer_num_workers=1 serialization)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.parse
from dataclasses import asdict

from audit_short_delta import ReqResult, make_text, send_chat, summarize


def make_multiline_text(seed: int, words: int, line_words: int = 18) -> str:
    toks = make_text(seed, words).split()
    return "\n".join(" ".join(toks[i : i + line_words]) for i in range(0, len(toks), line_words))


def build_20_turns(seed_base: int = 3000) -> list[dict]:
    msgs = [{"role": "system", "content": "You are a helpful coding assistant. Reply concisely."}]
    for t in range(10):
        u = make_multiline_text(seed_base + t * 2, words=2600)
        a = make_multiline_text(seed_base + t * 2 + 1, words=1300)
        msgs.append({"role": "user", "content": f"Turn {t} tool output:\n{u}"})
        msgs.append({"role": "assistant", "content": f"Turn {t} analysis:\n{a}"})
    return msgs


def run_multiturn_audit(url: str, model: str, out_path: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "10.0.1.71"
    port = parsed.port or 8000
    all_results: list[ReqResult] = []
    summaries: list[dict] = []

    base_turns = build_20_turns(3000)
    warm = send_chat(
        host, port, model,
        base_turns + [{"role": "user", "content": "Acknowledge warmup with OK."}],
        max_tokens=4, scenario="M-warmup-60k", idx=0,
    )
    print(
        f"M-warmup-60k: prompt={warm.prompt_tokens} cached={warm.cached_tokens} "
        f"client={warm.client_ttft_ms}ms gpu={warm.server_ttft_ms}ms "
        f"queue={warm.server_queue_ms}ms host={warm.host_overhead_ms}ms err={warm.error}"
    )
    all_results.append(warm)

    # 1. Solo multi-turn cached-delta ladder {128, 256, 512, 1024, 2048}
    for delta_words, label in [
        (80, "M-solo-d128"),
        (165, "M-solo-d256"),
        (335, "M-solo-d512"),
        (670, "M-solo-d1024"),
        (1340, "M-solo-d2048"),
    ]:
        rows = []
        for r in range(3):
            delta = f"New tool output {label}-{r}:\n" + make_multiline_text(
                4000 + delta_words * 10 + r, words=delta_words
            )
            rows.append(send_chat(
                host, port, model,
                base_turns + [{"role": "user", "content": delta}],
                max_tokens=8, scenario=label, idx=r,
            ))
        all_results.extend(rows)
        summaries.append(summarize(label, rows))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        # 2. Concurrent multi-turn waves (4 and 8, sync vs 25ms stagger)
        for conc, stagger_ms, delta_words, label in [
            (4, 0.0, 225, "M-wave4-d350-sync"),
            (4, 25.0, 225, "M-wave4-d350-stag25ms"),
            (8, 0.0, 225, "M-wave8-d350-sync"),
            (8, 25.0, 225, "M-wave8-d350-stag25ms"),
            (8, 0.0, 670, "M-wave8-d1024-sync"),
        ]:
            t_wave0 = time.perf_counter()
            futs = []
            for i in range(conc):
                delta = f"Subagent {label}-{i}:\n" + make_multiline_text(
                    6000 + conc * 100 + int(stagger_ms) * 10 + delta_words + i,
                    words=delta_words,
                )
                futs.append(pool.submit(
                    send_chat, host, port, model,
                    base_turns + [{"role": "user", "content": delta}],
                    8, label, i, i * stagger_ms,
                ))
            rows = [f.result() for f in futs]
            wave_ms = (time.perf_counter() - t_wave0) * 1000.0
            all_results.extend(rows)
            s = summarize(label, rows)
            s["wave_wall_ms"] = round(wave_ms, 1)
            print(f"  -> {label} wave_wall={wave_ms:.1f}ms")
            summaries.append(s)

        # 3. Short multi-turn delta during c4 active decode
        bg_futs = [
            pool.submit(
                send_chat, host, port, model,
                [{"role": "user", "content": f"Write a thorough 400-word explanation of topic {i}: " + make_text(9100 + i, 40)}],
                256, "M-bg-decode", i, 0.0,
            )
            for i in range(4)
        ]
        time.sleep(1.5)
        rows_d = []
        for r in range(4):
            delta = f"UnderDecode {r}:\n" + make_multiline_text(9500 + r, words=335)
            rows_d.append(send_chat(
                host, port, model,
                base_turns + [{"role": "user", "content": delta}],
                max_tokens=8, scenario="M-under-c4-decode-d512", idx=r,
            ))
            time.sleep(0.2)
        for f in bg_futs:
            f.result()
        all_results.extend(rows_d)
        summaries.append(summarize("M-under-c4-decode-d512", rows_d))

        # 4. Short multi-turn delta arriving during a cold 131K prefill!
        cold_131k = make_multiline_text(8888, words=86000)
        pf_fut = pool.submit(
            send_chat, host, port, model,
            [{"role": "user", "content": "Summarize:\n" + cold_131k}],
            8, "M-bg-131k-cold", 0, 0.0,
        )
        time.sleep(2.0)
        rows_p = []
        for r in range(4):
            delta = f"During131K {r}:\n" + make_multiline_text(9700 + r, words=335)
            rows_p.append(send_chat(
                host, port, model,
                base_turns + [{"role": "user", "content": delta}],
                max_tokens=8, scenario="M-under-131k-pf-d512", idx=r,
            ))
            time.sleep(1.5)
        pf_res = pf_fut.result()
        print(f"  -> M-bg-131k-cold: prompt={pf_res.prompt_tokens} client={pf_res.client_ttft_ms}ms gpu={pf_res.server_ttft_ms}ms host={pf_res.host_overhead_ms}ms")
        all_results.append(pf_res)
        all_results.extend(rows_p)
        summaries.append(summarize("M-under-131k-pf-d512", rows_p))

        # 5. Four concurrent cold 32K prompts (no newlines -> tests single-thread renderer vs multi-worker)
        t_c0 = time.perf_counter()
        cfuts = [
            pool.submit(
                send_chat, host, port, model,
                [{"role": "user", "content": f"Cold32K {i}: " + make_text(7100 + i, words=21000)}],
                8, "M-cold4-32k-sync", i, 0.0,
            )
            for i in range(4)
        ]
        crows = [f.result() for f in cfuts]
        cwall = (time.perf_counter() - t_c0) * 1000.0
        all_results.extend(crows)
        cs = summarize("M-cold4-32k-sync", crows)
        cs["wave_wall_ms"] = round(cwall, 1)
        print(f"  -> M-cold4-32k-sync wave_wall={cwall:.1f}ms")
        summaries.append(cs)

    with open(out_path, "w") as f:
        json.dump({"summaries": summaries, "results": [asdict(r) for r in all_results]}, f, indent=2)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.0.1.71:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--out", default="/tmp/multiturn_delta_audit.json")
    args = ap.parse_args()
    run_multiturn_audit(args.url, args.model, args.out)


if __name__ == "__main__":
    main()
