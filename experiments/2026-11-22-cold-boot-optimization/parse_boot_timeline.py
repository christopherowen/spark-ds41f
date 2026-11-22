#!/usr/bin/env python3
"""Parse docker logs --timestamps from a spark-ds41f container into a stage timeline."""

from __future__ import annotations

import datetime as dt
import re
import sys

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z\s+(.*)$")

MILESTONES: list[tuple[str, str]] = [
    ("api_banner", "non-default args:"),
    ("api_tok", "dsv41-structural-dsml-v1"),
    ("api_mm_done", "Multi-modal warmup completed"),
    ("engine_core_init", "Initializing a V1 LLM engine"),
    ("mp_executor_ready", "DP group leader:"),
    ("dist_init_start", "distributed_init_method="),
    ("pynccl_ready", "vLLM is using nccl=="),
    ("mp_groups_done", "assigned as DP rank"),
    ("load_scratch", "Loading model from scratch"),
    ("prealloc_pool", "Pre-allocated CUDA pool"),
    ("buddy_soak_target", "Buddy hugepage sponge soaked"),
    ("buddy_trim_target", "Trimmed buddy sponge"),
    ("weights_loaded_target", "Loading weights took"),
    ("moe_backend_target", "Using TileLangExperts"),
    ("packed_bf16_target", "Packed the BF16 LM head"),
    ("direct_done_target", "Direct loader: weights loaded"),
    ("dspark_loaded", "DSpark draft model loaded:"),
    ("model_load_done", "Model loading took"),
    ("kv_layout", "Using BLHNC KV cache layout"),
    ("l2_prefetch_ready", "[l2_prefetch] TileLang kernel ready"),
    ("jit_warmup_300", "JIT kernel warmup ("),
    ("tilelang_prep_done", "Prepared "),
    ("encoder_cache_init", "Encoder cache will be initialized"),
    ("kv_cache_done", "GPU KV cache size:"),
    ("jit_warmup_2", "JIT kernel warmup finished"),
    ("screened_vocab", "Using certified W_q screened"),
    ("capture_start", "Capturing model for DSpark speculator"),
    ("capture_done", "Graph capturing finished"),
    ("init_engine_done", "init engine (profile, create kv cache, warmup model) took"),
    ("server_ready", "Application startup complete."),
]


def parse_ts(s: str) -> float:
    # Truncate nanoseconds to microseconds
    head, frac = s.split(".")
    frac = (frac + "000000")[:6]
    t = dt.datetime.strptime(f"{head}.{frac}", "%Y-%m-%dT%H:%M:%S.%f")
    return t.replace(tzinfo=dt.timezone.utc).timestamp()


def main() -> None:
    lines = sys.stdin.read().splitlines()
    events: list[tuple[float, str, str]] = []
    t0: float | None = None
    seen: dict[str, int] = {}
    for line in lines:
        m = TS_RE.match(line)
        if not m:
            continue
        ts = parse_ts(m.group(1))
        msg = m.group(2)
        if t0 is None:
            t0 = ts
            events.append((ts, "first_log", msg[:90]))
        for key, pat in MILESTONES:
            if pat in msg:
                cnt = seen.get(key, 0) + 1
                seen[key] = cnt
                tag = key if cnt == 1 else f"{key}#{cnt}"
                events.append((ts, tag, msg[:100]))
                break

    if not events or t0 is None:
        print("No timestamped log lines found.")
        return

    print(f"{'t+s':>7s}  {'dt_s':>6s}  {'milestone':28s}  message")
    print("-" * 115)
    prev = t0
    for ts, tag, msg in events:
        print(f"{ts - t0:7.2f}  {ts - prev:6.2f}  {tag:28s}  {msg}")
        prev = ts


if __name__ == "__main__":
    main()
