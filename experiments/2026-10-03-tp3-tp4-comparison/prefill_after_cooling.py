"""Run the matched prefill/prefix suffix from a separately cooled start.

Use the normal `bin/spark bench` arguments, with suites prefill,prefix (or either
section separately), seed 0,
decode cases prose,code, concurrency 1,2,4,8 and min/max samples both 3.
Only the skipped decode suite's RNG draws are replayed. No decode GPU requests
are repeated. The normal identity checks, precooling, telemetry and aborts stay
active. This is a cold-start recovery screen, not a sustained-load pass.
"""

import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def advance_past_decode(rng, options):
    assert options.decode_cases == ["prose", "code"]
    assert options.concurrency == [1, 2, 4, 8]
    assert options.min_samples == options.max_samples == 3
    assert options.seed == 0
    points = [
        f"{case}-c{concurrency}"
        for case in options.decode_cases
        for concurrency in options.concurrency
    ]
    rng.sample(points, len(points))  # discarded warm-up order
    for _ in range(3):
        rng.shuffle(points)  # every point remains active until round three


def advance_past_prefill(rng, options):
    sizes = [size for size in options.prefill_sizes if size < options.max_model_len - 64]
    for _ in range(options.prefill_repeats):
        for _ in rng.sample(sizes, len(sizes)):
            rng.randrange(1 << 30)


def main():
    root = Path(__file__).resolve().parents[2]
    loader = importlib.machinery.SourceFileLoader("matched_bench", str(root / "bin/spark"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    original = module.BENCH_SUITE_FUNCTIONS["prefill"]
    original_prefix = module.BENCH_SUITE_FUNCTIONS["prefix"]

    def matched_prefill(bench, options, rng):
        assert options.suites in ("prefill,prefix", "prefill")
        advance_past_decode(rng, options)
        return original(bench, options, rng)

    def matched_prefix(bench, options, rng):
        assert options.suites in ("prefill,prefix", "prefix")
        if options.suites == "prefix":
            advance_past_decode(rng, options)
            advance_past_prefill(rng, options)
        return original_prefix(bench, options, rng)

    module.BENCH_SUITE_FUNCTIONS["prefill"] = matched_prefill
    module.BENCH_SUITE_FUNCTIONS["prefix"] = matched_prefix
    arguments = module.parser().parse_args()
    assert arguments.func is module.command_bench
    return arguments.func(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
