"""Run the matched prefill/prefix suffix from a separately cooled start.

Use the normal `bin/spark3 bench` arguments, with suites prefill,prefix, seed 0,
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


def main():
    root = Path(__file__).resolve().parents[2]
    loader = importlib.machinery.SourceFileLoader("matched_bench", str(root / "bin/spark3"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    original = module.BENCH_SUITE_FUNCTIONS["prefill"]

    def matched_prefill(bench, options, rng):
        assert options.suites == "prefill,prefix"
        advance_past_decode(rng, options)
        return original(bench, options, rng)

    module.BENCH_SUITE_FUNCTIONS["prefill"] = matched_prefill
    arguments = module.parser().parse_args()
    assert arguments.func is module.command_bench
    return arguments.func(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
