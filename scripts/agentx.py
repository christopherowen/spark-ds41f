#!/usr/bin/env python3
"""SemiAnalysis AgentX (agentic-coding trace replay) against the cluster, with the pinned harness.

usage (on the head node):
  scripts/agentx.py install                    (the pinned harness in its own venv)
  scripts/agentx.py fetch TOKENIZER            (between windows: download the pinned dataset)
  scripts/agentx.py prepare TOKENIZER --in-window
                                               (serving stopped: rebuild the corpus once for the replay)
  scripts/agentx.py stub [--port P]            (a model-free OpenAI streaming server)
  scripts/agentx.py run URL --model NAME TOKENIZER --concurrency N --out DIR [--duration S]
                        [--avoid-cpus LIST] [--require-prepared] [-- HARNESS ARGS]
  scripts/agentx.py summarize DIR              (rewrite DIR/agentx.json from a run's files)
  scripts/agentx.py table RUN LABEL...         (one row per point of results/private/agentx/RUN)
  where TOKENIZER is --tokenizer REPO --tokenizer-revision SHA [--hf-hub-cache DIR], the served
  checkpoint in the Hugging Face cache.

AgentX replays 393 recorded Claude Code sessions (a main agent plus up to 10 parallel
subagents, with their recorded gaps) as closed-loop sessions: --concurrency N keeps N session
trees alive. Prompt text is synthetic; lengths and prefix reuse are the recording's. The
harness (agentx-harness 1.0.7, a fork of NVIDIA AIPerf) warms each lane, then measures for
--duration seconds (AgentX: one hour) and stamps submission_valid.

The harness is pinned twice: by version, installed only from requirements/agentx-harness.txt
(every distribution's SHA-256) into a venv outside the deployment checkout, and by the dataset
revision, which fetch downloads and checks; everything after reads both offline. The first
replay of a corpus rebuilds every trace through the tokenizer, holding all of their block ids
(tens of GiB) for minutes, more memory than a serving node has free: prepare does it once with
serving stopped, against a local stub server, then checks that a second replay restores it within
a measurement's memory cap.

A run keeps the client off the serving cores (taskset over the cores the serving container is
not pinned to) and inside a memory-capped scope (systemd-run --user --scope -p MemoryMax=), so
a runaway client is killed instead of pushing the node under its memory guard. Beside the
harness's own files it samples the server's /metrics every --interval seconds
(server-series.jsonl) and writes agentx.json: the harness's verdicts and latency, the server's
prefix-cache hits, cached prompt tokens, preemptions and KV use over the profiling window, and
what an uncached prompt token cost.
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = "agentx-harness"
HARNESS_VERSION = "1.0.7"
REQUIREMENTS = ROOT / "requirements" / "agentx-harness.txt"
# Python the harness supports (3.11-3.13); uv provides one with headers (crick builds from source).
PYTHON = "3.12"
HOME = Path(os.environ.get("AGENTX_HOME", Path.home() / "spark-lab" / "agentx"))
VENV = HOME / f"venv-{HARNESS_VERSION}"
CACHE = HOME / "cache"  # the harness's reconstructed-dataset cache (AIPERF_DATASET_MMAP_CACHE_DIR)
# The corpora: the harness's --public-dataset name -> (Hugging Face dataset, the revision measured).
DATASETS = {
    "semianalysis_cc_traces_weka_062126": ("semianalysisai/cc-traces-weka-062126",
                                           "23f152f6f0f9399a85901b89a6458def0ef16729"),
    "semianalysis_cc_traces_weka_062126_256k": ("semianalysisai/cc-traces-weka-062126-256k",
                                                "8fecd2fc56694469f758f0afbbb6335ad3043740"),
}
DATASET = "semianalysis_cc_traces_weka_062126"
SEED = 42  # the agentx scenario's default seed; the replay cache is keyed by it
DURATION = 3600  # AgentX's measurement window
INTERVAL = 5.0  # seconds between /metrics samples (ours, and the harness's own collector)
# Load-generator processes: the harness's default follows the core count (up to 14 here); a fixed
# number keeps the client the same at every point. Each drives its share of lanes asynchronously.
WORKERS_MAX = 8
WARMUP_GRACE = 3600
MEMORY_MAX = "12G"  # a measurement's client (it restores the rebuilt corpus)
PREPARE_MEMORY_MAX = "64G"  # the one-time rebuild, with serving stopped
HOLDS = (Path.home() / "spark-hold.json", Path.home() / "spark3-hold.json")

# Server series kept in every sample: counters and gauges whole, histograms as sum and count,
# and the buckets of the few whose distribution the report reads.
SERIES_PREFIX = "vllm:"
BUCKETS_KEPT = ("vllm:request_prefill_kv_computed_tokens", "vllm:time_to_first_token_seconds",
                "vllm:request_queue_time_seconds", "vllm:request_prompt_tokens")
LABELS_KEPT = ("source", "position", "finished_reason", "le")


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + message, flush=True)


# ---------------------------------------------------------------- the harness command

def harness_command(url: str, model: str, tokenizer: str, revision: str, concurrency: int, artifacts: str,
                    duration: int = DURATION, seed: int = SEED, dataset: str = DATASET,
                    venv: Path = VENV, extra: list[str] | None = None, workers: int = WORKERS_MAX) -> list[str]:
    """The agentx profile command. Every scenario default stays (warmup, start ratios,
    cache-busting, idle caps, failure threshold); --tokenizer-trust-remote-code is never
    passed (DeepSeek's tokenizer is a plain tokenizer.json), nor --max-context-length (the
    server runs the model's full 1,048,576 tokens; an over-long request counts as an
    overflow, which the scenario limits to 1%). The tokenizer is the served checkpoint's,
    named by repository and revision: offline, the harness finds it in the hub cache only so."""
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset}; pinned: {', '.join(DATASETS)}")
    return [str(venv / "bin" / "agentx"), "profile",
            "--url", url, "--endpoint", "/v1/chat/completions",
            "--model", model, "--tokenizer", tokenizer, "--tokenizer-revision", revision,
            "--concurrency", str(concurrency),
            "--public-dataset", dataset, "--random-seed", str(seed),
            "--benchmark-duration", str(duration),
            "--output-artifact-dir", artifacts,
            "--server-metrics-formats", "json", "jsonl",
            "--workers-max", str(workers),
            # The preset bounds the warm-up barrier at 1,800 s; at 32 sessions the primers alone are
            # about 6M tokens of prefill (30-40 min here). A longer bound changes nothing measured.
            "--warmup-grace-period", str(WARMUP_GRACE),
            "--ui", "none",
            *(extra or [])]


def harness_environment(base: dict[str, str] | None = None, cache: Path = CACHE,
                        hub: str | None = None) -> dict[str, str]:
    """Offline (the dataset and tokenizer were fetched by prepare; ``hub`` is the Hugging Face
    cache holding them), the replay cache in the lab directory, a client keep-alive under
    vLLM's 5 s, and the harness's own /metrics collector at our interval rather than three
    times a second."""
    env = dict(os.environ if base is None else base)
    if hub:
        env["HF_HUB_CACHE"] = hub
    env.update({
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "AIPERF_DATASET_MMAP_CACHE_DIR": str(cache),
        "AIPERF_HTTP_KEEPALIVE_TIMEOUT": "4",
        "AIPERF_SERVER_METRICS_COLLECTION_INTERVAL": str(INTERVAL),
    })
    return env


def cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.replace(" ", "").split(","):
        if part:
            low, _, high = part.partition("-")
            cpus += range(int(low), int(high or low) + 1)
    return cpus


def client_cpus(avoid: str, available: set[int] | None = None) -> str:
    """The cores the client may use: every core this process may use but the serving
    container's (``avoid``, as docker's --cpuset-cpus)."""
    available = set(os.sched_getaffinity(0)) if available is None else available
    left = sorted(available - set(cpu_list(avoid)))
    if not left:
        raise SystemExit(f"no core left for the client outside {avoid}")
    return ",".join(map(str, left))


def confined(command: list[str], cpus: str | None, memory_max: str | None) -> list[str]:
    """``command`` on ``cpus`` inside a transient user scope capped at ``memory_max`` with no
    swap: the client is killed before it can take the node's free memory."""
    wrapped = list(command)
    if cpus:
        wrapped = ["taskset", "-c", cpus, *wrapped]
    if memory_max:
        wrapped = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={memory_max}",
                   "-p", "MemorySwapMax=0", *wrapped]
    return wrapped


# ---------------------------------------------------------------- /metrics sampling

def parse_metrics(text: str) -> dict[str, float]:
    """vLLM's Prometheus text as {series: value}: names under vllm:, summed over the labels
    not kept (engine, model_name), _created series and unkept buckets dropped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(SERIES_PREFIX):
            continue
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        if name.endswith("_created"):
            continue
        if name.endswith("_bucket") and name.removesuffix("_bucket") not in BUCKETS_KEPT:
            continue
        kept = sorted((key, label) for key, label in re.findall(r'(\w+)="([^"]*)"', labels)
                      if key in LABELS_KEPT)
        key = name + ("{" + ",".join(f'{k}="{v}"' for k, v in kept) + "}" if kept else "")
        try:
            number = float(value)
        except ValueError:
            continue
        if not math.isnan(number):
            out[key] = out.get(key, 0.0) + number
    return out


class Sampler:
    """Every ``interval`` seconds one JSON line {"t": epoch seconds, "m": series}; a failed
    scrape is a line with "error", never an exception."""

    def __init__(self, url: str, path: Path, interval: float = INTERVAL):
        self.url, self.path, self.interval = url.rstrip("/") + "/metrics", path, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self, handle) -> None:
        record: dict = {"t": round(time.time(), 3)}
        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                record["m"] = parse_metrics(response.read().decode("utf-8", errors="replace"))
        except OSError as error:
            record["error"] = str(error)[:200]
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()

    def loop(self) -> None:
        with self.path.open("a") as handle:
            while True:
                self.sample(handle)
                if self.stop.wait(self.interval):
                    self.sample(handle)
                    return

    def __enter__(self) -> "Sampler":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop.set()
        self.thread.join(timeout=30)
        return False


def read_series(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "m" in row:
            rows.append(row)
    return rows


def window(rows: list[dict], start: float, end: float) -> list[dict]:
    """The samples bracketing [start, end]: the last at or before start through the first at
    or after end (all of them when the window is not covered)."""
    if not rows:
        return []
    first = max((i for i, row in enumerate(rows) if row["t"] <= start), default=0)
    last = min((i for i, row in enumerate(rows) if row["t"] >= end), default=len(rows) - 1)
    return rows[first:last + 1]


def delta(rows: list[dict], name: str) -> float | None:
    """A counter's growth over the rows (summed over its label sets)."""
    if len(rows) < 2:
        return None
    total = lambda row: sum(v for k, v in row["m"].items() if k == name or k.startswith(name + "{"))
    return total(rows[-1]) - total(rows[0])


def gauge(rows: list[dict], name: str) -> dict | None:
    values = sorted(row["m"][name] for row in rows if name in row["m"])
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]
    return {"mean": round(statistics.fmean(values), 4), "p50": pick(0.5), "p90": pick(0.9), "max": values[-1]}


# ---------------------------------------------------------------- summaries

def harness_files(out: Path) -> tuple[Path | None, Path | None]:
    """The run's profile_export_aiperf.json and profile_export.jsonl (the harness may nest them)."""
    summary = next(iter(sorted(out.rglob("profile_export_aiperf.json"))), None)
    records = next(iter(sorted(out.rglob("profile_export.jsonl"))), None)
    return summary, records


def metric(data: dict, name: str, *stats: str) -> dict | float | None:
    entry = data.get(name)
    if not isinstance(entry, dict):
        return None
    if len(stats) == 1:
        return entry.get(stats[0])
    return {stat: entry.get(stat) for stat in stats if entry.get(stat) is not None} or None


def harness_summary(data: dict) -> dict:
    meta = data.get("metadata") or {}
    interactivity = metric(data, "output_token_throughput_per_user", "p10", "p50", "p90", "avg")
    return {
        "scenario": meta.get("scenario"),
        "submission_valid": meta.get("submission_valid"),
        "submission_invalid_reasons": meta.get("submission_invalid_reasons", []),
        "was_cancelled": data.get("was_cancelled"),
        "requests": metric(data, "request_count", "avg"),
        "errors": metric(data, "error_request_count", "avg") or 0,
        "context_overflows": metric(data, "context_overflow_count", "avg") or 0,
        "error_summary": data.get("error_summary") or [],
        "duration_s": metric(data, "benchmark_duration", "avg"),
        "ttft_ms": metric(data, "time_to_first_token", "p50", "p90", "p99", "avg"),
        "itl_ms": metric(data, "inter_token_latency", "p50", "p90", "p99", "avg"),
        "interactivity_tok_s": interactivity,
        "output_tok_s": metric(data, "output_token_throughput", "avg"),
        "requests_per_s": metric(data, "request_throughput", "avg"),
        "isl": metric(data, "input_sequence_length", "p50", "p90", "max", "avg"),
        "osl": metric(data, "output_sequence_length", "p50", "p90", "max", "avg"),
        "theoretical_prefix_cache_hit_pct": metric(data, "theoretical_prefix_cache_hit", "avg"),
        "usage_prompt_cache_read_pct": metric(data, "overall_usage_prompt_cache_read_pct", "avg"),
        "first_request_ns": metric(data, "min_request_timestamp", "avg"),
        "last_response_ns": metric(data, "max_response_timestamp", "avg"),
    }


def server_summary(rows: list[dict]) -> dict:
    """What the server did over the rows' span: prefix-cache hits, where prompt tokens came
    from, preemptions, KV use and queueing, and the prefill rate under this load."""
    if len(rows) < 2:
        return {"samples": len(rows)}
    queries, hits = delta(rows, "vllm:prefix_cache_queries_total"), delta(rows, "vllm:prefix_cache_hits_total")
    prompt, cached = delta(rows, "vllm:prompt_tokens_total"), delta(rows, "vllm:prompt_tokens_cached_total")
    computed = delta(rows, 'vllm:prompt_tokens_by_source_total{source="local_compute"}')
    prefill_s = delta(rows, "vllm:request_prefill_time_seconds_sum")
    prefills = delta(rows, "vllm:request_prefill_time_seconds_count")
    kv_computed = delta(rows, "vllm:request_prefill_kv_computed_tokens_sum")
    drafts = delta(rows, "vllm:spec_decode_num_drafts_total")
    ratio = lambda a, b, digits=4: round(a / b, digits) if a is not None and b else None
    return {
        "samples": len(rows),
        "seconds": round(rows[-1]["t"] - rows[0]["t"], 1),
        "requests_finished": delta(rows, "vllm:request_success_total"),
        "prefix_cache": {"queries": queries, "hits": hits, "hit_rate": ratio(hits, queries)},
        "prompt_tokens": {"total": prompt, "cached": cached, "cached_share": ratio(cached, prompt),
                          "computed": computed},
        "preemptions": delta(rows, "vllm:num_preemptions_total"),
        "kv_cache_usage": gauge(rows, "vllm:kv_cache_usage_perc"),
        "running": gauge(rows, "vllm:num_requests_running"),
        "waiting": gauge(rows, "vllm:num_requests_waiting"),
        "generation_tokens": delta(rows, "vllm:generation_tokens_total"),
        "prefill": {"requests": prefills, "seconds": prefill_s, "kv_computed_tokens": kv_computed,
                    # Computed prompt tokens per second of request prefill time: prefill time
                    # overlaps other requests' steps, so this is the rate one request sees.
                    "tok_per_request_s": ratio(kv_computed, prefill_s, 1)},
        "speculative": {
            "accepted_per_draft": ratio(delta(rows, "vllm:spec_decode_num_accepted_tokens_total"), drafts),
            "verified_per_draft": ratio(delta(rows, "vllm:spec_decode_num_draft_tokens_total"), drafts),
        },
    }


def records(path: Path | None, errors: bool = False) -> list[dict]:
    """Profiling-phase per-request records (those that completed without error unless ``errors``)."""
    if path is None or not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        meta = record.get("metadata") or {}
        if (record.get("error") and not errors) or str(meta.get("benchmark_phase", "profiling")).lower() != "profiling":
            continue
        out.append(record)
    return out


def profiling_window(rows: list[dict]) -> tuple[float, float] | None:
    """Epoch seconds from the first profiling request's start to the last one's end (the
    summary's own min/max timestamps are empty in harness 1.0.7)."""
    starts = [r["metadata"]["request_start_ns"] for r in rows if r.get("metadata", {}).get("request_start_ns")]
    ends = [r["metadata"]["request_end_ns"] for r in rows if r.get("metadata", {}).get("request_end_ns")]
    return (min(starts) / 1e9, max(ends) / 1e9) if starts and ends else None


def value(record: dict, name: str) -> float | None:
    entry = (record.get("metrics") or {}).get(name)
    if isinstance(entry, dict):
        entry = entry.get("value")
    return float(entry) if isinstance(entry, (int, float)) else None


def fit(points: list[tuple[float, float]]) -> dict | None:
    """Least squares y = a + b x with R²."""
    if len(points) < 3:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum((y - a - b * x) ** 2 for x, y in points) / ss if ss else None
    return {"n": len(points), "intercept_s": round(a, 4), "s_per_1k_tokens": round(1000 * b, 5),
            "r2": None if r2 is None else round(r2, 4)}


def miss_cost(rows: list[dict]) -> dict:
    """What a cache miss costs: time to first token against the prompt tokens the server did
    not have cached (usage.prompt_tokens_details, so the arm must report it), per request."""
    points, misses, prompts = [], [], []
    for record in rows:
        ttft, prompt = value(record, "time_to_first_token"), value(record, "usage_prompt_tokens")
        cached = value(record, "usage_prompt_cache_read_tokens")
        if ttft is None or prompt is None or cached is None:
            continue
        uncached = max(0.0, prompt - cached)
        points.append((uncached, ttft / 1000.0))
        misses.append(uncached)
        prompts.append(prompt)
    if not points:
        return {"requests": 0}
    return {
        "requests": len(points),
        "uncached_tokens": {"p50": statistics.median(misses), "p90": sorted(misses)[int(0.9 * (len(misses) - 1))],
                            "mean": round(statistics.fmean(misses), 1)},
        "prompt_tokens_p50": statistics.median(prompts),
        # Requests whose uncached part is more than half their prompt: a lost prefix, not a new tail.
        "misses_over_half": sum(1 for (m, _), p in zip(points, prompts) if p and m > p / 2),
        "ttft_vs_uncached": fit(points),
    }


def summarize(out: Path) -> dict:
    out = Path(out)
    summary_path, records_path = harness_files(out)
    data = json.loads(summary_path.read_text()) if summary_path else {}
    harness = harness_summary(data) if data else {}
    rows = read_series(out / "server-series.jsonl")
    start, end = harness.get("first_request_ns"), harness.get("last_response_ns")
    span = (start / 1e9, end / 1e9) if start and end else profiling_window(records(records_path, errors=True))
    if span:
        harness["profiling_window_utc"] = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) for t in span]
    result = {
        "harness": harness,
        "server_profiling": server_summary(window(rows, *span) if span else []),
        "server_whole_run": server_summary(rows),
        "miss_cost": miss_cost(records(records_path)),
    }
    run = out / "run.json"
    if run.exists():
        result["run"] = json.loads(run.read_text())
    (out / "agentx.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------------------------------------------------------------- run

def cgroup_memory_peak(pid: int) -> int | None:
    """memory.peak of the cgroup ``pid`` runs in (the client's scope), or None."""
    try:
        line = next(l for l in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if l.startswith("0::"))
        return int(Path("/sys/fs/cgroup" + line[3:].strip(), "memory.peak").read_text())
    except (OSError, StopIteration, ValueError):
        return None


def command_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if (out / "harness").exists() and any((out / "harness").iterdir()):
        raise SystemExit(f"{out}/harness is not empty; a point is never measured twice into one directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.require_prepared and not prepared_marker(args.dataset, args.seed).exists():
        raise SystemExit(f"{args.dataset} seed {args.seed} was never prepared on this host "
                         f"({prepared_marker(args.dataset, args.seed)}); run scripts/agentx.py prepare first")
    command = harness_command(args.url, args.model, args.tokenizer, args.tokenizer_revision, args.concurrency,
                              str(out / "harness"),
                              args.duration, args.seed, args.dataset, Path(args.venv), args.extra)
    cpus = client_cpus(args.avoid_cpus) if args.avoid_cpus else None
    wrapped = confined(command, cpus, None if args.memory_max == "none" else args.memory_max)
    run = {"command": wrapped, "concurrency": args.concurrency, "duration": args.duration, "seed": args.seed,
           "dataset": args.dataset, "dataset_revision": DATASETS[args.dataset][1], "harness": HARNESS_VERSION,
           "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
           "client_cpus": cpus, "memory_max": args.memory_max, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    log("agentx " + " ".join(command[1:]))
    peak = None
    with Sampler(args.url, out / "server-series.jsonl", args.interval), \
            (out / "harness.log").open("w") as handle:
        process = subprocess.Popen(wrapped, stdout=handle, stderr=subprocess.STDOUT,
                                   env=harness_environment(cache=Path(args.cache), hub=args.hf_hub_cache),
                                   start_new_session=True)
        try:
            while process.poll() is None:
                peak = cgroup_memory_peak(process.pid) or peak
                time.sleep(5)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)  # the harness writes its partial results on one SIGINT
            process.wait()
    run.update(exit=process.returncode, client_memory_peak_bytes=peak,
               ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    result = summarize(out)
    harness = result["harness"]
    log(f"agentx exit {process.returncode}: submission_valid {harness.get('submission_valid')}, "
        f"{harness.get('requests')} requests, {harness.get('errors')} errors, prefix-cache hit rate "
        f"{result['server_profiling'].get('prefix_cache', {}).get('hit_rate')}")
    print(json.dumps({"summary": result}, separators=(",", ":")), flush=True)
    return process.returncode


# ---------------------------------------------------------------- install and prepare

def command_install(args: argparse.Namespace) -> int:
    """uv's Python (with headers: crick builds from source), a fresh venv outside the deployment
    checkout, and only the hashed requirements (no resolution, no unpinned dependency)."""
    venv = Path(args.venv)
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if venv.exists():
        raise SystemExit(f"{venv} exists; remove it to reinstall")
    subprocess.run([uv, "python", "install", PYTHON], check=True)
    subprocess.run([uv, "venv", "--python", PYTHON, "--managed-python", str(venv)], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), "--require-hashes",
                    "--no-deps", "-r", str(REQUIREMENTS)], check=True)
    version = subprocess.run([str(venv / "bin" / "agentx"), "--version"], text=True, capture_output=True)
    log(f"installed: agentx {version.stdout.strip()} in {venv}")
    return 0 if HARNESS_VERSION in version.stdout else 1


def prepared_marker(dataset: str, seed: int) -> Path:
    return HOME / f"prepared-{dataset}-seed{seed}.json"


def fetched_paths(dataset: str, hub: str) -> list[Path]:
    """What fetch leaves: the pinned snapshot's traces in the hub cache and the datasets
    library's cache built from it (an offline load needs both)."""
    repo, revision = DATASETS[dataset]
    org, name = repo.split("/")
    return [Path(hub) / f"datasets--{org}--{name}" / "snapshots" / revision / "traces.jsonl",
            Path(hub).parent / "datasets" / f"{org}___{name}" / "default" / "0.0.0" / revision]


def held() -> str | None:
    for path in HOLDS:
        if path.exists():
            return str(path)
    return None


def dataset_snapshot(dataset: str, venv: Path = VENV, offline: bool = True, hub: str | None = None,
                     cpus: str | None = None, memory_max: str | None = None) -> Path:
    """The dataset's snapshot in the Hugging Face cache, downloaded (main) unless offline; it
    must be the pinned revision. Online, it is then loaded once the way the harness loads it
    (datasets builds its own cache, which an offline load needs) and loaded again offline."""
    repo, revision = DATASETS[dataset]
    python = [str(venv / "bin" / "python"), "-I", "-c"]
    snapshot_code = ("import sys; from huggingface_hub import snapshot_download as s; "
                     "print(s(repo_id=sys.argv[1], repo_type='dataset', revision='main'))")
    # The harness's call (aiperf/dataset/loader/base_hf_dataset.py), so the cache it builds is the one found.
    load_code = ("import sys; from datasets import load_dataset as l; "
                 "d = l(sys.argv[1], name=None, split='train', trust_remote_code=False, streaming=False); "
                 "print(len(d))")

    def call(code: str, offline_now: bool) -> str:
        env = dict(os.environ, **({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"} if offline_now else {}),
                   **({"HF_HUB_CACHE": hub} if hub else {}))
        process = subprocess.run(confined([*python, code, repo], cpus, memory_max), text=True,
                                 capture_output=True, env=env, cwd=Path.home())
        if process.returncode:
            raise SystemExit(f"{repo}: {process.stderr.strip().splitlines()[-1] if process.stderr.strip() else 'failed'}")
        return process.stdout.strip().splitlines()[-1]

    snapshot = Path(call(snapshot_code, offline))
    if snapshot.name != revision:
        raise SystemExit(f"{repo} main is {snapshot.name}, not the pinned {revision}: a new upload; "
                         "pin it deliberately (DATASETS) before measuring on it")
    if not offline:
        rows = call(load_code, False)
        if call(load_code, True) != rows:
            raise SystemExit(f"{repo}: the offline load differs from the online one")
        log(f"{repo}: {rows} traces, loadable offline")
    return snapshot


class _Stub(http.server.BaseHTTPRequestHandler):
    """Answers chat completions at once: one content chunk and the usage the client asked for."""
    protocol_version = "HTTP/1.1"
    model = "stub"

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            self._send(200, json.dumps({"object": "list", "data": [{"id": self.model, "object": "model"}]}).encode())
        elif self.path.startswith("/metrics"):
            self._send(200, b"", "text/plain")
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 1)
        prompt = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4
        usage = {"prompt_tokens": prompt, "completion_tokens": tokens, "total_tokens": prompt + tokens,
                 # As vLLM reports with --enable-prompt-tokens-details (made up: nine tenths cached).
                 "prompt_tokens_details": {"cached_tokens": prompt * 9 // 10}}
        stamp = int(time.time())
        if not body.get("stream"):
            message = {"role": "assistant", "content": "x " * tokens}
            self._send(200, json.dumps({"id": "stub", "object": "chat.completion", "created": stamp,
                                        "model": self.model, "usage": usage,
                                        "choices": [{"index": 0, "message": message, "finish_reason": "length"}]}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def event(data: str) -> None:
            payload = f"data: {data}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")

        chunk = lambda delta, finish=None: json.dumps({"id": "stub", "object": "chat.completion.chunk", "created": stamp,
                                                        "model": self.model, "choices": [{"index": 0, "delta": delta,
                                                                                          "finish_reason": finish}]})
        event(chunk({"role": "assistant", "content": "x"}))
        event(chunk({"content": " x" * max(0, tokens - 1)}, "length"))
        event(json.dumps({"id": "stub", "object": "chat.completion.chunk", "created": stamp, "model": self.model,
                          "choices": [], "usage": usage}))
        event("[DONE]")
        self.wfile.write(b"0\r\n\r\n")


def stub_server(port: int, model: str) -> http.server.ThreadingHTTPServer:
    handler = type("Stub", (_Stub,), {"model": model})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def free_port() -> int:
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def command_stub(args: argparse.Namespace) -> int:
    server = stub_server(args.port, args.model)
    log(f"stub serving {args.model} on 127.0.0.1:{server.server_address[1]}")
    server.serve_forever()
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    """Download the pinned dataset and build the datasets library's cache from it, so every
    later step reads it offline. A download: between windows only, never under a hold."""
    if held():
        raise SystemExit(f"{held()} exists: someone is measuring; fetch between windows")
    free = shutil.disk_usage(Path.home()).free / 2 ** 30
    if free < args.min_free_gib:
        raise SystemExit(f"{free:.0f} GiB free under {Path.home()}; fetch needs {args.min_free_gib}")
    cpus = client_cpus(args.avoid_cpus) if args.avoid_cpus else None
    snapshot = dataset_snapshot(args.dataset, Path(args.venv), offline=False, hub=args.hf_hub_cache, cpus=cpus,
                                memory_max=None if args.memory_max == "none" else args.memory_max)
    log(f"{args.dataset}: {snapshot}")
    return 0


def memory_available() -> int | None:
    try:
        line = next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
        return int(line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return None


def gib(text: str) -> float:
    units = {"K": 2 ** -20, "M": 2 ** -10, "G": 1, "T": 2 ** 10}
    return float(text[:-1]) * units[text[-1].upper()] if text[-1].upper() in units else float(text) / 2 ** 30


def stub_replay(args: argparse.Namespace, out: Path, memory_max: str | None, cpus: str | None,
                concurrency: int = 2) -> tuple[int, int | None, float]:
    """A short replay against a local stub: the harness configures the dataset exactly as a
    measurement does (the scenario's duration rule broken on purpose: marked invalid, discarded)."""
    server = stub_server(free_port(), args.model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    command = harness_command(url, args.model, args.tokenizer, args.tokenizer_revision, concurrency,
                              str(out / "harness"), 60,
                              args.seed, args.dataset, Path(args.venv), ["--unsafe-override", "--no-server-metrics"])
    out.mkdir(parents=True, exist_ok=True)
    started, peak = time.time(), None
    try:
        with (out / "harness.log").open("w") as handle:
            process = subprocess.Popen(confined(command, cpus, memory_max), stdout=handle, stderr=subprocess.STDOUT,
                                       env=harness_environment(cache=Path(args.cache), hub=args.hf_hub_cache),
                                       start_new_session=True)
            while process.poll() is None:
                peak = cgroup_memory_peak(process.pid) or peak
                time.sleep(2)
    finally:
        server.shutdown()
        server.server_close()
    return process.returncode, peak, round(time.time() - started, 1)


def command_prepare(args: argparse.Namespace) -> int:
    """Rebuild the fetched dataset for the replay, once per dataset, seed and tokenizer, so a
    measurement restores it in seconds: a first replay tokenizes every trace and holds every
    trace's block ids (about 340 million) in memory, tens of GiB, more than a serving node has
    free. So it runs with serving stopped, inside our own window (--in-window, as lab.py's agentx
    job does), and refuses unless the memory it may use is free. A second replay must then
    restore the cache and stay under the measurement's cap (--run-memory-max)."""
    hold = held()
    if hold and not args.in_window:
        raise SystemExit(f"{hold} exists: someone is measuring; prepare runs in our own window (--in-window)")
    if args.in_window:
        holder = json.loads(Path(hold).read_text()).get("holder", "") if hold else ""
        if not holder.startswith("spark-lab"):
            raise SystemExit(f"--in-window, but the cluster is not held by a lab window ({hold or 'no hold'}: "
                             f"{holder or '-'})")
    memory_max = None if args.memory_max == "none" else args.memory_max
    available = memory_available()
    if memory_max and available is not None and available / 2 ** 30 < gib(memory_max) + 8:
        raise SystemExit(f"{available / 2 ** 30:.0f} GiB available; the rebuild may use {memory_max} and needs 8 "
                         "GiB beside it: stop serving first (prepare runs inside a lab window)")
    snapshot = dataset_snapshot(args.dataset, Path(args.venv), offline=True, hub=args.hf_hub_cache)
    cpus = client_cpus(args.avoid_cpus) if args.avoid_cpus else None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = HOME / "prepare" / f"{args.dataset}-seed{args.seed}-{stamp}"
    code, peak, seconds = stub_replay(args, base / "rebuild", memory_max, cpus)
    if code or harness_files(base / "rebuild")[0] is None:
        log(f"the rebuild failed (exit {code}, peak {(peak or 0) / 2 ** 30:.1f} GiB); see {base / 'rebuild' / 'harness.log'}")
        return 1
    log(f"rebuilt {args.dataset} seed {args.seed} in {seconds} s (peak {(peak or 0) / 2 ** 30:.1f} GiB)")
    # The restore at the largest point's lane count, with the measurement's workers and cap.
    code, hit_peak, hit_seconds = stub_replay(args, base / "restore", args.run_memory_max, cpus, args.restore_concurrency)
    logs = [base / "restore" / "harness.log", *sorted((base / "restore").rglob("aiperf.log"))]
    restored = any("dataset cache HIT" in path.read_text(errors="replace") for path in logs if path.exists())
    if code or not restored:
        log(f"the second replay did not restore the cache (exit {code}, hit {restored}); see {base / 'restore'}")
        return 1
    record = {"dataset": args.dataset, "revision": DATASETS[args.dataset][1], "snapshot": str(snapshot),
              "seed": args.seed, "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
              "harness": HARNESS_VERSION, "cache": args.cache,
              "rebuild": {"seconds": seconds, "memory_peak_bytes": peak, "memory_max": memory_max},
              "restore": {"seconds": hit_seconds, "memory_peak_bytes": hit_peak, "memory_max": args.run_memory_max},
              "replays": str(base), "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    prepared_marker(args.dataset, args.seed).write_text(json.dumps(record, indent=2) + "\n")
    log(f"prepared: the restore peaked at {(hit_peak or 0) / 2 ** 30:.1f} GiB "
        f"(cap {args.run_memory_max}): {prepared_marker(args.dataset, args.seed)}")
    return 0


# ---------------------------------------------------------------- table

def fmt(value, digits: int = 0, scale: float = 1.0) -> str:
    if value is None:
        return "-"
    return f"{value * scale:.{digits}f}"


def table(run: str, labels: list[str], base: Path | None = None) -> str:
    base = base or ROOT / "results" / "private" / "agentx"
    lines = [f"AgentX {run}: one row per point (profiling window; server counters over the same span)",
             f"{'point':<18}{'valid':>6}{'reqs':>7}{'errs':>6}{'TTFT p50/p90 s':>17}{'tok/s/user p50/p90':>20}"
             f"{'out tok/s':>10}{'hit%':>7}{'ideal%':>8}{'cached%':>8}{'preempt':>8}{'KV p90/max':>12}"
             f"{'wait mean/max':>14}{'s/1k miss':>10}{'peak GiB':>9}"]
    for label in labels:
        path = base / run / label / "agentx.json"
        if not path.exists():
            lines.append(f"{label:<18} (no result)")
            continue
        data = json.loads(path.read_text())
        h, s = data.get("harness", {}), data.get("server_profiling", {})
        ttft, inter = h.get("ttft_ms") or {}, h.get("interactivity_tok_s") or {}
        kv, wait = s.get("kv_cache_usage") or {}, s.get("waiting") or {}
        cost = (data.get("miss_cost") or {}).get("ttft_vs_uncached") or {}
        peak = (data.get("run") or {}).get("client_memory_peak_bytes")
        lines.append(
            f"{label:<18}{str(h.get('submission_valid'))[:5]:>6}{fmt(h.get('requests')):>7}{fmt(h.get('errors')):>6}"
            f"{fmt(ttft.get('p50'), 2, 1e-3) + '/' + fmt(ttft.get('p90'), 2, 1e-3):>17}"
            f"{fmt(inter.get('p50'), 1) + '/' + fmt(inter.get('p90'), 1):>20}{fmt(h.get('output_tok_s'), 1):>10}"
            f"{fmt((s.get('prefix_cache') or {}).get('hit_rate'), 1, 100):>7}"
            f"{fmt(h.get('theoretical_prefix_cache_hit_pct'), 1):>8}"
            f"{fmt((s.get('prompt_tokens') or {}).get('cached_share'), 1, 100):>8}{fmt(s.get('preemptions')):>8}"
            f"{fmt(kv.get('p90'), 2) + '/' + fmt(kv.get('max'), 2):>12}"
            f"{fmt(wait.get('mean'), 1) + '/' + fmt(wait.get('max')):>14}{fmt(cost.get('s_per_1k_tokens'), 3):>10}"
            f"{fmt(peak, 1, 2 ** -30):>9}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    def client(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--venv", default=str(VENV))
        sub.add_argument("--cache", default=str(CACHE))
        sub.add_argument("--dataset", default=DATASET, choices=sorted(DATASETS))
        sub.add_argument("--seed", type=int, default=SEED)
        sub.add_argument("--tokenizer", required=True, help="the served checkpoint's repository (org/name)")
        sub.add_argument("--tokenizer-revision", required=True, help="the served checkpoint's revision")
        sub.add_argument("--hf-hub-cache", help="the Hugging Face hub cache holding it and the dataset")
        sub.add_argument("--avoid-cpus", help="the serving container's cores (docker --cpuset-cpus)")
        sub.add_argument("--memory-max", default=MEMORY_MAX, help="the client's memory cap ('none': no scope)")

    install = commands.add_parser("install")
    install.add_argument("--venv", default=str(VENV))
    fetch = commands.add_parser("fetch")
    client(fetch)
    fetch.add_argument("--min-free-gib", type=float, default=20.0)
    prepare = commands.add_parser("prepare")
    client(prepare)
    prepare.set_defaults(memory_max=PREPARE_MEMORY_MAX)
    prepare.add_argument("--model", default="deepseek-v4.1-flash")
    prepare.add_argument("--run-memory-max", default=MEMORY_MAX, help="the measurement's cap the restore must fit")
    prepare.add_argument("--in-window", action="store_true", help="inside our own lab window (serving stopped)")
    prepare.add_argument("--restore-concurrency", type=int, default=16)
    stub = commands.add_parser("stub")
    stub.add_argument("--port", type=int, default=0)
    stub.add_argument("--model", default="deepseek-v4.1-flash")
    run = commands.add_parser("run")
    run.add_argument("url")
    client(run)
    run.add_argument("--model", required=True, help="the served model name")
    run.add_argument("--concurrency", type=int, required=True)
    run.add_argument("--duration", type=int, default=DURATION)
    run.add_argument("--out", required=True)
    run.add_argument("--interval", type=float, default=INTERVAL)
    run.add_argument("--require-prepared", action="store_true",
                     help="refuse unless prepare rebuilt this dataset and seed here")
    summary = commands.add_parser("summarize")
    summary.add_argument("out")
    shown = commands.add_parser("table")
    shown.add_argument("run")
    shown.add_argument("labels", nargs="+")
    return root


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    extra = tokens[tokens.index("--") + 1:] if "--" in tokens else []  # run: further harness arguments
    args = parser().parse_args(tokens[:tokens.index("--")] if "--" in tokens else tokens)
    args.extra = extra
    if args.command == "install":
        return command_install(args)
    if args.command == "fetch":
        return command_fetch(args)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "stub":
        return command_stub(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "summarize":
        print(json.dumps(summarize(Path(args.out)), indent=2))
        return 0
    print(table(args.run, args.labels), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
