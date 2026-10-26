"""servebench command line: ``run`` measures a live server, ``engine`` times its steps,
``model`` records what it computes, ``prefill`` times cold prompts, ``suite`` runs a fixed
set of them, ``compare`` judges two results (or two suite runs)."""

import argparse
import datetime
import gzip
import json
import os
import sys
import time
import urllib.request

from . import SCHEMA, __version__, behaviour, engine, load, prefill, prompts, report, stall, suite


def model_id(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0]["id"]


def write_json(path: str, value: dict) -> None:
    """Pretty JSON, gzip-compressed when the path ends in .gz."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, indent=None if path.endswith(".gz") else 1)


def read_json(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def command_run(args: argparse.Namespace) -> int:
    prompt_set = prompts.load(args.prompts)
    spec = load.Load(mode=args.mode, concurrency=args.concurrency, rate=args.rate,
                     warmup_s=args.warmup, duration_s=args.duration, batches=args.batches,
                     max_tokens=args.max_tokens, fixed_length=not args.natural_length,
                     seed=args.seed, context_tokens=args.context)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    model = args.model or model_id(args.url)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    raw = load.run(args.url, model, prompt_set, spec, vllm=not args.no_vllm, extra_body=extra)
    summary = report.summarize(raw, slo_ttft_s=args.slo_ttft, slo_tpot_ms=args.slo_tpot)
    result = {
        "schema": SCHEMA, "tool": {"name": "servebench", "version": __version__},
        "started_utc": started, "label": args.label,
        "server": {"url": args.url, "model": model, "adapter": "openai" if args.no_vllm else "vllm"},
        "workload": spec.describe() | {"prompts": args.prompts, "prompt_count": len(prompt_set),
                                       "prompts_sha256": prompts.digest(prompt_set),
                                       "extra_body": extra},
        "summary": summary,
    }
    if raw.get("documents"):
        result["documents"] = raw["documents"]
    if args.save_requests:
        origin = raw["start"]
        result["requests"] = [
            {key: (round(value - origin, 4) if key in ("sent", "first", "end") and value is not None else value)
             for key, value in record.items() if key != "events"}
            | {"events": [[round(t - origin, 4), n] for t, n in record["events"]]}
            for record in raw["records"]]
    if args.output:
        write_json(args.output, result)
    tps = summary.get("output_tps", {})
    line = {"summary": f"servebench {spec.mode}", "label": args.label, "mode": spec.mode,
            "concurrency": spec.concurrency if spec.mode != "poisson" else None,
            "rate": spec.rate if spec.mode == "poisson" else None,
            "tps_mean": tps.get("mean"), "ci95_pct": tps.get("ci95_pct"), "windows": tps.get("n"),
            "steps_per_s": summary.get("server", {}).get("steps_per_s", {}).get("mean"),
            "tokens_per_step": summary.get("server", {}).get("output_tokens_per_step", {}).get("mean"),
            "context_tokens": spec.context_tokens or None,
            "prefix_cache_hit_rate": summary.get("server", {}).get("prefix_cache_hit_rate", {}).get("mean"),
            "ttft_p50_s": summary["latency"]["ttft_s"].get("p50"),
            "tpot_p50_ms": summary["latency"]["tpot_ms"].get("p50"),
            "failed": summary["requests"]["failed"],
            "foreign_requests": summary["requests"]["foreign"], "output": args.output}
    print(json.dumps(line), flush=True)
    if summary["requests"]["foreign"]:
        print(f"servebench: {summary['requests']['foreign']} foreign requests finished during the run: "
              "the server was not serving this run alone", file=sys.stderr, flush=True)
        return 3
    return 1 if summary["requests"]["failed"] else 0


def command_engine(args: argparse.Namespace) -> int:
    if args.grid == "default":
        longest = args.max_context or engine.max_model_len(args.url)
        shapes = engine.default_shapes(drafts=args.drafts, max_context=longest, prefill=args.prefill,
                                       max_requests=args.max_requests)
    else:
        with open(args.grid, encoding="utf-8") as handle:
            shapes = json.load(handle)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    raw = engine.run(args.url, shapes, replays=args.replays, warmup=args.warmup, settle=args.settle)
    rows = engine.summarize(raw)
    result = {"schema": engine.SCHEMA, "tool": {"name": "servebench", "version": __version__},
              "started_utc": started, "label": args.label,
              "server": {"url": args.url, "workers": raw["workers"], "rank": raw["rank"],
                         "prefix_cache_reset": raw["prefix_cache_reset"]},
              "workload": {"grid": args.grid, "replays": args.replays, "warmup": args.warmup,
                           "settle": raw["settle"], "drafts": args.drafts, "prefill": args.prefill,
                           "max_requests": args.max_requests},
              "engine": rows}
    if args.output:
        write_json(args.output, result)
    for row in rows:
        d = row["step_ms"]
        print(f"{row['label']:32s} {d['mean']:9.3f} ms ±{d['ci95_pct'] or 0:.2f}%", file=sys.stderr)
    widest = max((row["step_ms"].get("ci95_pct") or 0) for row in rows)
    print(json.dumps({"summary": "servebench engine", "label": args.label, "shapes": len(rows),
                      "widest_ci95_pct": widest, "prefix_cache_reset": raw["prefix_cache_reset"],
                      "output": args.output}), flush=True)
    return 0 if raw["prefix_cache_reset"] else 1


def command_model(args: argparse.Namespace) -> int:
    prompt_set = prompts.load(args.prompts)[:args.limit or None]
    extra = json.loads(args.extra_body) if args.extra_body else {}
    model = args.model or model_id(args.url)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    generated = behaviour.generate(args.url, model, prompt_set, concurrency=args.concurrency,
                                   max_tokens=args.max_tokens, fixed_length=args.fixed_length,
                                   extra_body=extra)
    sequences, source = generated["sequences"], "self"
    if args.corpus:
        sequences, source = read_json(args.corpus)["generation"]["sequences"], args.corpus
    scores, score_seconds = None, None
    if not args.no_score:
        began = time.monotonic()
        scores = behaviour.score(args.url, model, sequences, top=args.top, concurrency=args.score_concurrency)
        score_seconds = round(time.monotonic() - began, 2)
    result = {"schema": behaviour.SCHEMA, "tool": {"name": "servebench", "version": __version__},
              "started_utc": started, "label": args.label, "server": {"url": args.url, "model": model},
              "workload": {"prompts": args.prompts, "prompt_count": len(prompt_set),
                           "prompts_sha256": prompts.digest(prompt_set), "concurrency": args.concurrency,
                           "max_tokens": args.max_tokens, "fixed_length": args.fixed_length,
                           "top": args.top, "extra_body": extra},
              "generation": generated,
              "corpus": {"source": source, "sha256": behaviour.corpus_digest(sequences),
                         "sequences": sum(1 for s in sequences if not s.get("error"))},
              "scores": scores, "score_seconds": score_seconds,
              "self_agreement": behaviour.self_agreement(scores) if scores and source == "self" else None}
    if args.output:
        write_json(args.output, result)
    failed = sum(1 for s in generated["sequences"] + (scores or []) if s.get("error"))
    agreement = result["self_agreement"] or {}
    print(json.dumps({"summary": "servebench model", "label": args.label,
                      "acceptance_length": generated["acceptance"].get("acceptance_length"),
                      "outputs_sha256": behaviour.corpus_digest(generated["sequences"]),
                      "rank1_fraction": agreement.get("rank1_fraction"),
                      "perplexity": agreement.get("perplexity"), "generate_s": generated["seconds"],
                      "score_s": score_seconds, "failed": failed, "output": args.output}), flush=True)
    return 1 if failed else 0


def compare_results(base: dict, candidate: dict, tolerance: float) -> tuple[str, dict]:
    schema = base.get("schema")
    if schema != candidate.get("schema"):
        raise SystemExit(f"different result kinds: {schema} vs {candidate.get('schema')}")
    if schema == behaviour.SCHEMA:
        return schema, behaviour.compare(base, candidate)
    if schema == engine.SCHEMA:
        return schema, engine.compare(base, candidate, tolerance_pct=tolerance)
    if schema == prefill.SCHEMA:
        return schema, prefill.compare(base["prefill"], candidate["prefill"], tolerance_pct=tolerance)
    if schema == stall.SCHEMA:
        return schema, stall.compare(base["stall"], candidate["stall"], tolerance_pct=tolerance)
    return schema, report.compare(base, candidate, tolerance_pct=tolerance)


def format_result(schema: str, result: dict, labels: list[str]) -> str:
    if schema == behaviour.SCHEMA:
        return behaviour.format_comparison(result, labels[0], labels[1])
    return report.format_comparison(result, labels[0][:12], labels[1][:12])


def command_compare(args: argparse.Namespace) -> int:
    if os.path.isdir(args.base) and os.path.isdir(args.candidate):
        result = suite.compare(read_json, args.base, args.candidate,
                               lambda b, c: compare_results(b, c, args.tolerance))
        labels = [label or path for label, path in zip(result["labels"], (args.base, args.candidate))]
        print(f"suite {result['suite']} ({result['suite_sha256'][:12]}): {labels[0]} -> {labels[1]}")
        for step, outcome in result["steps"].items():
            print(f"\n== {step}")
            print(f"skipped: {outcome['skipped']}" if "skipped" in outcome else format_result(*outcome, labels))
        if args.json:
            print(json.dumps(result))
        return 0
    runs = [read_json(path) for path in (args.base, args.candidate)]
    labels = [run.get("label") or path for run, path in zip(runs, (args.base, args.candidate))]
    schema, result = compare_results(runs[0], runs[1], args.tolerance)
    print(format_result(schema, result, labels))
    if args.json:
        print(json.dumps(result))
    return 0


def command_spread(args: argparse.Namespace) -> int:
    runs = [read_json(path) for path in args.grids]
    against = [read_json(path) for path in args.against] if args.against else None
    result = engine.spread(runs, against)
    print(engine.format_spread(result))
    if args.json:
        print(json.dumps(result))
    return 0


def command_prefill(args: argparse.Namespace) -> int:
    model = args.model or model_id(args.url)
    longest = engine.max_model_len(args.url)
    sizes = [int(size) for size in args.sizes.split(",")]
    skipped = [size for size in sizes if longest and size > longest - 1024]
    extra = json.loads(args.extra_body) if args.extra_body else {}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    samples = prefill.run(args.url, model, [s for s in sizes if s not in skipped], repeats=args.repeats,
                          vllm=not args.no_vllm, extra_body=extra,
                          log=lambda message: print(message, file=sys.stderr, flush=True))
    rows = prefill.summarize(samples)
    result = {"schema": prefill.SCHEMA, "tool": {"name": "servebench", "version": __version__},
              "started_utc": started, "label": args.label,
              "server": {"url": args.url, "model": model, "max_model_len": longest},
              "workload": {"sizes": sizes, "repeats": args.repeats, "skipped": skipped, "extra_body": extra,
                           "question": prefill.QUESTION},
              "prefill": rows, "samples": samples}
    if args.output:
        write_json(args.output, result)
    failed = sum(row["failed"] for row in rows)
    print(json.dumps({"summary": "servebench prefill", "label": args.label,
                      "tps": {row["size"]: row["tps"].get("mean") for row in rows},
                      "ci95_pct": {row["size"]: row["tps"].get("ci95_pct") for row in rows},
                      "skipped": skipped, "failed": failed, "output": args.output}), flush=True)
    return 1 if failed else 0


def command_stall(args: argparse.Namespace) -> int:
    model = args.model or model_id(args.url)
    longest = engine.max_model_len(args.url)
    sizes = [int(size) for size in args.sizes.split(",")]
    skipped = [size for size in sizes if longest and size > longest - 1024]
    extra = json.loads(args.extra_body) if args.extra_body else {}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    probes = tuple(float(offset) for offset in args.probes.split(",")) if args.probes else ()
    samples = stall.run(args.url, model, [s for s in sizes if s not in skipped], streams=args.streams,
                        max_tokens=args.max_tokens, warmup_s=args.warmup, tail_s=args.tail, repeats=args.repeats,
                        vllm=not args.no_vllm, extra_body=extra, probes=probes, probe_tokens=args.probe_tokens,
                        probe_max_tokens=args.probe_max_tokens,
                        log=lambda message: print(message, file=sys.stderr, flush=True))
    rows = stall.summarize(samples)
    result = {"schema": stall.SCHEMA, "tool": {"name": "servebench", "version": __version__},
              "started_utc": started, "label": args.label,
              "server": {"url": args.url, "model": model, "max_model_len": longest},
              "workload": {"sizes": sizes, "streams": args.streams, "max_tokens": args.max_tokens,
                           "warmup_s": args.warmup, "tail_s": args.tail, "repeats": args.repeats, "skipped": skipped,
                           "probes": list(probes), "probe_tokens": args.probe_tokens,
                           "probe_max_tokens": args.probe_max_tokens,
                           "extra_body": extra, "question": prefill.QUESTION},
              "stall": rows, "samples": samples}
    if args.output:
        write_json(args.output, result)
    failed = sum(row["failed"] for row in rows)
    print(json.dumps({"summary": "servebench stall", "label": args.label,
                      "stall_s": {row["size"]: row["stall_s"].get("mean") for row in rows},
                      "during_p50_s": {row["size"]: row["during_p50_s"].get("mean") for row in rows},
                      "before_p50_s": {row["size"]: row["before_p50_s"].get("mean") for row in rows},
                      "tps": {row["size"]: row["tps"].get("mean") for row in rows},
                      "skipped": skipped, "failed": failed, "output": args.output}), flush=True)
    return 1 if failed else 0


def command_suite(args: argparse.Namespace) -> int:
    definition = suite.load(args.suite)
    metadata = None
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as handle:
            metadata = json.load(handle)
    index = suite.run(main, args.url, definition, args.out, label=args.label, corpus=args.corpus,
                      metadata=metadata, log=lambda message: print(message, file=sys.stderr, flush=True))
    failed = [step["id"] for step in index["steps"] if step["exit"] != 0]
    print(json.dumps({"summary": "servebench suite", "suite": definition["name"],
                      "suite_sha256": index["suite_sha256"][:12], "label": args.label, "out": args.out,
                      "seconds": {step["id"]: step["seconds"] for step in index["steps"]}, "failed": failed}),
          flush=True)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """The command line, without running anything (lab tools validate arguments with it)."""
    parser = argparse.ArgumentParser(prog="servebench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="measure a live OpenAI-compatible server")
    run.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    run.add_argument("--mode", choices=("closed", "poisson", "burst"), default="closed")
    run.add_argument("--concurrency", type=int, default=8, help="closed and burst: requests in flight")
    run.add_argument("--rate", type=float, default=1.0, help="poisson: requests per second")
    run.add_argument("--warmup", type=float, default=60.0, help="seconds before the window")
    run.add_argument("--duration", type=float, default=300.0, help="window length in seconds")
    run.add_argument("--batches", type=int, default=10, help="sub-windows for the intervals")
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--natural-length", action="store_true",
                     help="let requests stop at end of sequence (default: exactly --max-tokens each)")
    run.add_argument("--prompts", default="default", help="prompt set name or JSONL path")
    run.add_argument("--context", type=int, default=0, metavar="TOKENS",
                     help="closed: each stream keeps its own cached document of about TOKENS tokens "
                          "before every question (decode at depth)")
    run.add_argument("--seed", type=int, default=0, help="orders the prompts and the arrivals")
    run.add_argument("--model", help="model id (default: the server's first)")
    run.add_argument("--extra-body", help="JSON merged into every request body")
    run.add_argument("--no-vllm", action="store_true",
                     help="plain OpenAI requests (no cache_salt, ignore_eos, per-chunk usage)")
    run.add_argument("--slo-ttft", type=float, help="goodput: time to first token limit, seconds")
    run.add_argument("--slo-tpot", type=float, help="goodput: time per output token limit, ms")
    run.add_argument("--label", help="name for tables")
    run.add_argument("--output", help="write the full result JSON here")
    run.add_argument("--save-requests", action="store_true", help="keep every request's timeline")
    run.set_defaults(func=command_run)
    grid = sub.add_parser("engine", help="time identical dummy steps inside the server (engine cost grid)")
    grid.add_argument("url", help="base URL of a server with the dev-mode collective RPC")
    grid.add_argument("--grid", default="default", help="'default' or a JSON file of shapes")
    grid.add_argument("--replays", type=int, default=100, help="timed rounds over every shape")
    grid.add_argument("--warmup", type=int, default=2, help="untimed runs per shape first")
    grid.add_argument("--settle", type=int, default=1,
                      help="untimed runs of the same shape before each timed one (switching shapes "
                           "rebuilds the dummy metadata, the harness's cost)")
    grid.add_argument("--prefill", action="store_true",
                      help="add 8K prefill chunks (dummy prefill does not track real prefill)")
    grid.add_argument("--drafts", type=int, default=5, help="drafts per request in verify rows")
    grid.add_argument("--max-context", type=int, help="longest context (default: the server's)")
    grid.add_argument("--max-requests", type=int, default=16,
                      help="leave out shapes of more requests than this (the server's --max-num-seqs)")
    grid.add_argument("--label", help="name for tables")
    grid.add_argument("--output", help="write the result JSON here")
    grid.set_defaults(func=command_engine)
    behave = sub.add_parser("model", help="model behaviour: greedy acceptance by draft position and "
                                          "teacher-forced log-probabilities")
    behave.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    behave.add_argument("--corpus", help="score this earlier model run's sequences (default: this run's own)")
    behave.add_argument("--prompts", default="default", help="prompt set name or JSONL path")
    behave.add_argument("--limit", type=int, help="first N prompts only")
    behave.add_argument("--concurrency", type=int, default=16, help="greedy requests in flight")
    behave.add_argument("--max-tokens", type=int, default=256)
    behave.add_argument("--fixed-length", action="store_true",
                        help="write exactly --max-tokens past end of sequence (default: stop there)")
    behave.add_argument("--top", type=int, default=20,
                        help="log-probabilities kept per position (vLLM's --max-logprobs, default 20)")
    behave.add_argument("--score-concurrency", type=int, default=8)
    behave.add_argument("--no-score", action="store_true", help="generation and acceptance only")
    behave.add_argument("--model", help="model id (default: the server's first)")
    behave.add_argument("--extra-body", help="JSON merged into every generation request body")
    behave.add_argument("--label", help="name for tables")
    behave.add_argument("--output", help="write the result JSON here (.gz to compress)")
    behave.set_defaults(func=command_model)
    spreads = sub.add_parser("spread", help="boot-to-boot spread of engine grids (one grid per boot)")
    spreads.add_argument("grids", nargs="+", help="engine results of one build, each from its own boot")
    spreads.add_argument("--against", nargs="+", help="another set to compare, boots as the samples")
    spreads.add_argument("--json", action="store_true")
    spreads.set_defaults(func=command_spread)
    cold = sub.add_parser("prefill", help="cold prefill rate on real-text prompts of fixed sizes")
    cold.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    cold.add_argument("--sizes", default="32768,262144", help="prompt sizes in tokens, comma-separated")
    cold.add_argument("--repeats", type=int, default=3)
    cold.add_argument("--model", help="model id (default: the server's first)")
    cold.add_argument("--extra-body", help="JSON merged into every request body")
    cold.add_argument("--no-vllm", action="store_true", help="plain OpenAI requests (no cache salt)")
    cold.add_argument("--label", help="name for tables")
    cold.add_argument("--output", help="write the result JSON here")
    cold.set_defaults(func=command_prefill)
    waits = sub.add_parser("stall", help="decode stalls of concurrent streams while a long cold prompt prefills")
    waits.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    waits.add_argument("--sizes", default="131072", help="document sizes in tokens, comma-separated")
    waits.add_argument("--streams", type=int, default=8, help="decoding streams")
    waits.add_argument("--max-tokens", type=int, default=16384,
                       help="each stream's output budget; it must outlast warm-up and prefill")
    waits.add_argument("--warmup", type=float, default=20.0, help="seconds of decoding before the document")
    waits.add_argument("--tail", type=float, default=2.0, help="seconds of decoding after its first token")
    waits.add_argument("--probes", default="",
                       help="seconds after the document to send short cold requests, comma-separated "
                            "(their time to first token while it prefills); leave them free slots")
    waits.add_argument("--probe-tokens", type=int, default=500, help="each probe's prompt size in tokens")
    waits.add_argument("--probe-max-tokens", type=int, default=64, help="each probe's output budget")
    waits.add_argument("--repeats", type=int, default=3)
    waits.add_argument("--model", help="model id (default: the server's first)")
    waits.add_argument("--extra-body", help="JSON merged into every request body")
    waits.add_argument("--no-vllm", action="store_true", help="plain OpenAI requests (no cache salt)")
    waits.add_argument("--label", help="name for tables")
    waits.add_argument("--output", help="write the result JSON here")
    waits.set_defaults(func=command_stall)
    fixed = sub.add_parser("suite", help="run a fixed set of measurements (servebench/suites/*.json)")
    fixed.add_argument("url", help="base URL, e.g. http://127.0.0.1:8000")
    fixed.add_argument("--suite", default="screen", help="built-in suite name or JSON path")
    fixed.add_argument("--out", required=True, help="directory for every step's result and suite.json")
    fixed.add_argument("--corpus", help="model steps score this model result's text (the first arm's)")
    fixed.add_argument("--label", help="name for tables")
    fixed.add_argument("--metadata", metavar="JSON",
                       help="a JSON file recorded in suite.json as given (the deployment's identity)")
    fixed.set_defaults(func=command_suite)
    compare = sub.add_parser("compare", help="judge a candidate run against a base run")
    compare.add_argument("base", help="a result file, or a suite run's directory")
    compare.add_argument("candidate")
    compare.add_argument("--tolerance", type=float, default=3.0,
                         help="budget in %% for metrics without an interval")
    compare.add_argument("--json", action="store_true", help="also print the comparison as JSON")
    compare.set_defaults(func=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
